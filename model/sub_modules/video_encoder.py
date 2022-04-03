import torch
from torch import nn
import torch.nn.functional as F


def _construct_conv_layers(input_dim, hidden_dim, kernel_size, stride, padding):
    layers = []
    for layer_idx in range(len(kernel_size)):
        in_dim = input_dim if layer_idx == 0 else hidden_dim
        out_dim = hidden_dim
        layers.append(nn.Conv1d(in_dim, out_dim, kernel_size[layer_idx],
                                stride[layer_idx], padding[layer_idx]))
        layers.append(nn.GELU())
        layers.append(nn.BatchNorm1d(out_dim))
    return nn.Sequential(*layers)


class VideoInitializer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.init_conv = _construct_conv_layers(input_dim, hidden_dim, kernel_size, stride, padding)

    def forward(self, visual_feat):
        return self.init_conv(visual_feat.transpose(-2, -1)).transpose(-2, -1)


class VideoFusionInitializer(nn.Module):
    def __init__(self, frame_dim, motion_dim, hidden_dim, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.frame_conv = nn.Sequential(
            nn.Conv1d(frame_dim, hidden_dim, kernel_size[0], stride[0], padding[0]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.motion_conv = nn.Sequential(
            nn.Conv1d(motion_dim, hidden_dim, kernel_size[0], stride[0], padding[0]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, visual_feat):
        frame_raw_feat, motion_raw_feat = torch.chunk(visual_feat, dim=-1, chunks=2)
        frame_feat = self.frame_conv(frame_raw_feat.transpose(-2, -1)).transpose(-2, -1)
        motion_feat = self.motion_conv(motion_raw_feat.transpose(-2, -1)).transpose(-2, -1)
        return F.gelu(frame_feat + motion_feat) - (frame_feat - motion_feat) ** 2


class VideoSeparateInitializer(nn.Module):
    def __init__(self, frame_dim, motion_dim, hidden_dim, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.frame_conv = nn.Sequential(
            nn.Conv1d(frame_dim, hidden_dim, kernel_size[0], stride[0], padding[0]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.motion_conv = nn.Sequential(
            nn.Conv1d(motion_dim, hidden_dim, kernel_size[0], stride[0], padding[0]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, visual_feat):
        frame_raw_feat, motion_raw_feat = torch.chunk(visual_feat, dim=-1, chunks=2)
        frame_feat = self.frame_conv(frame_raw_feat.transpose(-2, -1)).transpose(-2, -1)
        motion_feat = self.motion_conv(motion_raw_feat.transpose(-2, -1)).transpose(-2, -1)
        return frame_feat, motion_feat