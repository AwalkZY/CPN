from torch import nn


class LayerNormResidualBlock(nn.Module):
    def __init__(self, layer, input_dim, drop_ratio, input_position=0):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.norm = nn.LayerNorm(input_dim)
        self.input_position = input_position

    def forward(self, *inputs):
        if self.layer is None:
            return inputs[self.input_position]
        return self.norm(inputs[self.input_position] + self.dropout(self.layer(*inputs)))
