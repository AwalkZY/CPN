import torch
from torch import nn

from model.sub_modules.segment_tree.ensemble import EnsembleSegmentTree
from model.sub_modules.text_encoder import TextInitializer
from model.sub_modules.video_encoder import VideoSeparateInitializer, VideoFusionInitializer
from modules.attention_layers import TanhAttention


class QuestionAnswerModel(nn.Module):
    def __init__(self, model_dim, chunk_num, use_negative, text_config, video_config, tree_config, **kwargs):
        super().__init__()
        self.chunk_num = chunk_num
        self.model_dim = model_dim
        self.use_negative = use_negative
        self.text_encoder = TextInitializer(**text_config)
        self.video_encoder = VideoFusionInitializer(**video_config)
        self.segment_tree = EnsembleSegmentTree(**tree_config)
        self.core_dim = model_dim // chunk_num
        self.video_text_attn = TanhAttention(self.core_dim)
        self.quad_ffn = nn.Linear(4 * self.core_dim, self.core_dim)
        self.double_ffn = nn.Linear(2 * self.core_dim, self.core_dim)
        self.mix_up = nn.Linear(2 * self.core_dim, 1)

    def cqa_fusion(self, visual_feat, textual_feat, textual_mask):
        matrix_a, attn_logit = self.video_text_attn(visual_feat, textual_feat, textual_mask)
        # attn_logit in (bs, vis_len, tex_len)
        video_text_attn, text_video_attn = attn_logit.softmax(-1), attn_logit.softmax(1)
        matrix_b = video_text_attn.bmm(text_video_attn.transpose(-2, -1)).bmm(visual_feat)
        fusion = torch.cat((visual_feat, matrix_a, visual_feat * matrix_a,
                            visual_feat * matrix_b), dim=-1)
        # (bs, max_len, 1)
        return self.quad_ffn(fusion)

    def attn_fusion(self, visual_feat, textual_feat, textual_mask):
        matrix_a, attn_logit = self.video_text_attn(visual_feat, textual_feat, textual_mask)
        fusion = torch.cat((visual_feat, matrix_a), dim=-1)
        return self.double_ffn(fusion)

    def fuse_and_route(self, visual_feat, textual_feat, textual_mask):
        batch_size, visual_len, _ = visual_feat.size()
        textual_len = textual_feat.size(1)
        visual_feat = self.video_encoder(visual_feat)
        chunked_visual_feat = torch.stack(visual_feat.chunk(chunks=self.chunk_num, dim=-1),
                                          dim=1).view(-1, visual_len, self.model_dim // self.chunk_num)
        textual_feat, textual_agg = self.text_encoder(textual_feat, textual_mask)
        chunked_textual_feat = torch.stack(textual_feat.chunk(chunks=self.chunk_num, dim=-1),
                                           dim=1).view(-1, textual_len, self.model_dim // self.chunk_num)
        chunked_textual_agg = torch.stack(textual_agg.chunk(chunks=self.chunk_num, dim=-1),
                                          dim=1).view(-1, self.model_dim // self.chunk_num)
        chunked_textual_mask = textual_mask.unsqueeze(1).repeat(1, self.chunk_num, 1).view(-1, textual_len)
        # all in (bs * chunk_num, max_len, model_dim // chunk_num)
        compact_fusion = self.cqa_fusion(chunked_visual_feat, chunked_textual_feat, chunked_textual_mask)
        predict_result, raw_result = self.segment_tree(compact_fusion, chunked_textual_agg)
        predict_result = predict_result.view(batch_size, self.chunk_num, *predict_result.size()[1:])
        raw_result = raw_result.view(batch_size, self.chunk_num, *raw_result.size()[1:])
        return predict_result, raw_result

    def forward(self, visual_feat, textual_feat, textual_mask):
        all_result, raw_result = self.fuse_and_route(visual_feat, textual_feat, textual_mask)
        batch_size, chunk_num, max_len, class_num = raw_result.size()
        return {
            "avg_result": all_result.mean(dim=1),
            "all_result": raw_result.view(batch_size, -1, class_num)
        }
