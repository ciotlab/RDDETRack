import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


def positional_encoding_sine(num_embedding, d_model, max_num_embedding, normalize, scale):
    seq_embed = torch.arange(1, num_embedding+1)
    if normalize:
        eps = 1e-6
        if scale is None:
            scale = 2 * math.pi * max_num_embedding
        seq_embed = seq_embed / (seq_embed[-1] + eps) * scale
    dim_t = torch.arange(d_model)
    dim_t = max_num_embedding ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / d_model)

    pe = seq_embed[:, None] / dim_t
    pe = torch.stack((pe[:, 0::2].sin(), pe[:, 1::2].cos()), dim=2).flatten(1)
    return pe


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x