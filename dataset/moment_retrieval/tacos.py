import os

import h5py
import numpy as np

from dataset.moment_retrieval.base import BaseDataset, build_collate_data


class TACoS(BaseDataset):
    def __init__(self, data_path_template, vocab_path, max_frame_num, max_word_num, frame_dim, word_dim,
                 feature_path, action, **kwargs):
        super().__init__(data_path_template.format(action), vocab_path, max_frame_num, max_word_num)
        self.feature_path = feature_path
        self.collate_fn = build_collate_data(max_frame_num, max_word_num, frame_dim, word_dim)
        self.video_pool = {}

    def _load_frame_features(self, vid):
        if vid in self.video_pool:
            return self.video_pool[vid]
        else:
            features = np.load(os.path.join(self.feature_path, '%s.npy' % vid[:-4])).astype(np.float32)
            self.video_pool[vid] = features
            return features

    def collate_data(self, samples):
        return self.collate_fn(samples)
