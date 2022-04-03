from torch import nn
from modules import DynamicGRU


class TextInitializer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = DynamicGRU(input_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, textual_feat, textual_mask):
        textual_len = textual_mask.sum(-1) if textual_mask is not None else None
        return self.gru(textual_feat, textual_len)

