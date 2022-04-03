import os
import sys

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

from utils.container import resourceContainer
from utils.helper import sequence_mask


def build_collate_data(max_frame_num, max_word_num, action):
    def collate_fn(samples):
        batch = {'raw': [sample['raw'] for sample in samples]}
        visual_raw_feat = torch.stack([sample['visual_feat'] for sample in samples], dim=0)
        # (bsz, frame_len, visual_dim)
        frame_idx = torch.round(1.0 * torch.arange(0, max_frame_num + 1) / max_frame_num * visual_raw_feat.size(1)).long()
        frame_idx[frame_idx >= visual_raw_feat.size(1)] = visual_raw_feat.size(1) - 1
        visual_feat = torch.zeros(visual_raw_feat.size(0), max_frame_num, visual_raw_feat.size(-1))
        for idx in range(max_frame_num):
            s, e = frame_idx[idx], frame_idx[idx + 1]
            if s == e:
                visual_feat[:, idx] = visual_raw_feat[:, s]
            else:
                visual_feat[:, idx] = visual_raw_feat[:, s:e].mean(dim=1)
        textual_feat = torch.stack([F.pad(sample['question_embed'],
                                          [0, 0, 0, max_word_num - len(sample['question_embed'])])
                                    for sample in samples], dim=0)
        textual_len = torch.stack([sample['question_len'] for sample in samples], dim=0)
        answer_idx = torch.stack([sample['answer_encode'] for sample in samples], dim=0) if action == "train" else None
        textual_mask = sequence_mask(textual_len, max_word_num)
        batch.update({
            'net_input': {
                'visual_feat': visual_feat,
                'textual_feat': textual_feat,
                'textual_mask': textual_mask
            },
            'target': {
                'answer': answer_idx
            }
        })
        return batch

    return collate_fn


class MSVDQA(Dataset):
    def __init__(self, data_path, max_frame_num, action, min_word_num, max_word_num, **kwargs):
        super().__init__()
        assert action in ["train", "val", "test"], "Invalid Action Name!"
        self.action = action
        self.max_frame_num = max_frame_num
        self.min_word_num = min_word_num
        self.max_word_num = max_word_num
        self.video_feature = resourceContainer.fetch_resource("video_feature")
        if self.video_feature is None:
            self.video_feature = h5py.File((os.path.join(data_path, 'video_feature_20.h5')), "r")
            resourceContainer.save_resource("video_feature", self.video_feature)
        self.vocab = resourceContainer.fetch_resource("word_embedding")
        if self.vocab is None:
            self.vocab = torch.from_numpy(np.load(os.path.join(data_path, "word_embedding.npy")))
            resourceContainer.save_resource("word_embedding", self.vocab)
        self.raw_data = pd.read_json(os.path.join(data_path, '{}_qa_encode.json'.format(action)))
        self.bucket_data = self._init_train_data() if action == "train" else self.raw_data

    def _init_train_data(self):
        self.raw_data['question_length'] = self.raw_data.apply(lambda row: len(row['question'].split()), axis=1)
        bucket_data = self.raw_data[(self.raw_data['question_length'] >= self.min_word_num) &
                                    (self.raw_data['question_length'] <= self.max_word_num)]
        bucket_data = bucket_data.sample(frac=1)
        return bucket_data
        # 2-6; 7-11; 12-16; 17-21

    def __getitem__(self, index):
        example_id = self.bucket_data.iloc[index]['id']
        question_encode = self.bucket_data.iloc[index]['question_encode']
        question_id = torch.tensor([int(x) for x in question_encode.split(',')])
        question_len = torch.tensor(len(question_id))
        question_embed = F.embedding(question_id, self.vocab)
        answer_encode = torch.tensor(self.bucket_data.iloc[index]['answer_encode']) if self.action == "train" else None
        answer_raw = self.bucket_data.iloc[index]['answer']
        video_id = torch.tensor(self.bucket_data.iloc[index]['video_id'])
        vgg_feat = torch.tensor(self.video_feature['vgg'][video_id - 1])
        c3d_feat = torch.tensor(self.video_feature['c3d'][video_id - 1])
        visual_feat = torch.cat((vgg_feat, c3d_feat), dim=-1)
        return {
            "visual_feat": visual_feat,
            "question_embed": question_embed,
            "answer_encode": answer_encode,
            "question_len": question_len,
            "raw": [answer_raw, example_id]
        }

    def __len__(self):
        return self.bucket_data.shape[0]

    def collate_data(self, samples):
        return build_collate_data(self.max_frame_num, self.max_word_num, self.action)(samples)
