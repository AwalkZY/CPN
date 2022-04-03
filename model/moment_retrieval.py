import math

from torch import nn
import torch
import torch.nn.functional as F
from model.sub_modules.segment_tree.competitive import CompetitiveSegmentTree
from model.sub_modules.segment_tree.ensemble import EnsembleSegmentTree
from model.sub_modules.text_encoder import TextInitializer
from model.sub_modules.video_encoder import VideoInitializer
from modules.attention_layers import TanhAttention
import threading


class MomentRetrievalModel(nn.Module):
    def __init__(self, model_dim, chunk_num, use_negative, text_config, video_config, tree_config, **kwargs):
        super().__init__()
        self.chunk_num = chunk_num
        self.model_dim = model_dim
        self.use_negative = use_negative
        self.text_encoder = TextInitializer(**text_config)
        self.video_encoder = VideoInitializer(**video_config)
        self.segment_tree = CompetitiveSegmentTree(**tree_config)
        self.core_dim = model_dim // chunk_num
        self.video_text_attn = TanhAttention(self.core_dim)
        self.feed_forward = nn.Linear(4 * self.core_dim, self.core_dim)

    # def parallel_calc_boundary(self, compact_fusion, textual_agg):
    #     lock = threading.Lock()
    #     results = []
    #     exceptions = []
    #
    #     def _forward(inputs, module):
    #         try:
    #             output = module(*inputs)
    #             with lock:
    #                 results.append(output)
    #         except Exception as e:
    #             with lock:
    #                 exceptions.append(e)
    #
    #     threads = [threading.Thread(target=_forward,
    #                                 args=((compact_fusion[:, chunk_idx], textual_agg[:, chunk_idx]),
    #                                       self.segment_tree[chunk_idx]))
    #                for chunk_idx in range(self.chunk_num)]
    #     for thread in threads:
    #         thread.start()
    #     for thread in threads:
    #         thread.join()
    #     if exceptions:
    #         raise exceptions[0]
    #     boundary_logit = torch.stack(results, dim=1)
    #     return boundary_logit

    def fuse_and_route(self, visual_feat, textual_feat, textual_mask):
        batch_size, visual_len, _ = visual_feat.size()
        textual_len = textual_feat.size(1)
        visual_feat = self.video_encoder(visual_feat)
        textual_feat, textual_agg = self.text_encoder(textual_feat, textual_mask)
        chunked_visual_feat = torch.stack(visual_feat.chunk(chunks=self.chunk_num, dim=-1),
                                          dim=1).view(-1, visual_len, self.model_dim // self.chunk_num)
        chunked_textual_feat = torch.stack(textual_feat.chunk(chunks=self.chunk_num, dim=-1),
                                           dim=1).view(-1, textual_len, self.model_dim // self.chunk_num)
        chunked_textual_agg = torch.stack(textual_agg.chunk(chunks=self.chunk_num, dim=-1),
                                          dim=1).view(-1, self.model_dim // self.chunk_num)
        chunked_textual_mask = textual_mask.unsqueeze(1).repeat(1, self.chunk_num, 1).view(-1, textual_len)
        # all in (bs * chunk_num, max_len, model_dim // chunk_num)
        matrix_a, attn_logit = self.video_text_attn(chunked_visual_feat, chunked_textual_feat, chunked_textual_mask)
        # attn_logit in (bs, vis_len, tex_len)
        video_text_attn, text_video_attn = attn_logit.softmax(-1), attn_logit.softmax(1)
        matrix_b = video_text_attn.bmm(text_video_attn.transpose(-2, -1)).bmm(chunked_visual_feat)
        fusion = torch.cat((chunked_visual_feat, matrix_a, chunked_visual_feat * matrix_a,
                            chunked_visual_feat * matrix_b), dim=-1)
        # (bs, max_len, 1)
        compact_fusion = self.feed_forward(fusion)
        boundary_prob, inside_prob = self.segment_tree(compact_fusion, chunked_textual_agg)
        boundary_prob = boundary_prob.view(batch_size, self.chunk_num, *boundary_prob.size()[1:])
        inside_prob = inside_prob.view(batch_size, self.chunk_num, *inside_prob.size()[1:])
        return boundary_prob, inside_prob

    def forward(self, visual_feat, textual_feat, textual_mask):
        real_boundary_prob, real_inside_prob = self.fuse_and_route(visual_feat, textual_feat, textual_mask)
        # (bs, chunk_num, 1->max_len, 1->task_num)
        avg_boundary_prob = real_boundary_prob.mean(dim=1)
        avg_inside_prob = real_inside_prob.mean(dim=1)
        # assert (avg_boundary_prob.sum(dim=1).allclose(torch.tensor(1.0))), "Invalid Probability!"
        if not self.use_negative:
            return {
                "real_start": avg_boundary_prob[:, :, 0],
                "real_end": avg_boundary_prob[:, :, 1],
                "fake_start": None,
                "fake_end": None,
                "all_start": real_boundary_prob[:, :, :, 0],
                "all_end": real_boundary_prob[:, :, :, 1],
                "inside_prob": avg_inside_prob,
                "order_pred": None,
                "order_target": None
            }
        else:
            batch_size = visual_feat.size(0)
            idx = list(reversed(range(batch_size)))
            fake_textual_feat, fake_textual_mask = textual_feat[idx], textual_mask[idx]
            fake_boundary_prob, fake_root_feat = self.fuse_and_route(visual_feat, fake_textual_feat,
                                                                     fake_textual_mask)
            return None
