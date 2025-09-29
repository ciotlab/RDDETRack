import copy
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import logging
from einops import rearrange
from data_processing.dataset import get_dataset_and_dataloader
from models.misc_model import positional_encoding_sine


class Transformer(nn.Module):
    def __init__(self, d_model=512, n_head=8, num_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", return_intermediate=False, pre_ln=True):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        layer = TransformerLayer(d_model, n_head, dim_feedforward, dropout, activation, pre_ln)
        self.layers = _get_clones(layer, num_layers)
        self.return_intermediate = return_intermediate
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, source, query_pos, source_pos, key_padding_mask, targets):
        num_batch = source.shape[0]
        query = query.expand(num_batch, -1, -1)
        query_pos = query_pos.expand(num_batch, -1, -1)
        if source_pos is not None:
            source_pos = source_pos.expand(num_batch, -1, -1)
        intermediate_output = []
        intermediate_attn = []
        x = query
        attn = torch.empty(0)
        if 'track_query_hs_embed' in targets:
            track_query = targets['track_query_hs_embed']
            x = torch.cat([track_query, x], dim=1)
            track_query_pos = torch.zeros_like(track_query)
            query_pos = torch.cat([track_query_pos, query_pos], dim=1)
            query_mask = targets['track_query_mask']
        else:
            query_mask = None
        for i, layer in enumerate(self.layers):
            x, attn = layer(query=x, source=source, query_pos=query_pos, source_pos=source_pos,
                            key_padding_mask=key_padding_mask,
                            query_mask=query_mask)
            if self.return_intermediate and i < len(self.layers)-1:
                intermediate_output.append(x)
                intermediate_attn.append(attn)
        return x, attn, intermediate_output, intermediate_attn


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1, activation="relu", pre_ln=True):
        super().__init__()
        self.pre_ln = pre_ln
        # Multihead attention modules
        self.self_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query, source, query_pos, source_pos, key_padding_mask, query_mask=None):
        if self.pre_ln:
            x, cross_attn_map = self.forward_pre_ln(query, source, query_pos, source_pos, key_padding_mask, query_mask)
        else:
            x, cross_attn_map = self.forward_post_ln(query, source, query_pos, source_pos, key_padding_mask, query_mask)
        return x, cross_attn_map

    def forward_post_ln(self, query, source, query_pos, source_pos, key_padding_mask, query_mask):
        x = query
        q = k = self.with_pos_embed(x, query_pos)
        x2, self_attn_map = self.self_attn(query=q, key=k, value=x, key_padding_mask=query_mask)
        x = x + self.dropout1(x2)
        if query_mask is not None:
            x = x.masked_fill(query_mask.unsqueeze(-1), 0.0)
        x = self.norm1(x)
        x2, cross_attn_map = self.cross_attn(query=self.with_pos_embed(x, query_pos),
                                             key=self.with_pos_embed(source, source_pos), value=source,
                                             key_padding_mask=key_padding_mask)
        x = x + self.dropout2(x2)
        if query_mask is not None:
            x = x.masked_fill(query_mask.unsqueeze(-1), 0.0)
        x = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        if query_mask is not None:
            x = x.masked_fill(query_mask.unsqueeze(-1), 0.0)
        return x, cross_attn_map

    def forward_pre_ln(self, query, source, query_pos, source_pos, key_padding_mask, query_mask):
        x = query
        # self attention
        x2 = self.norm1(x)
        q = k = self.with_pos_embed(x2, query_pos)
        x2, self_attn_map = self.self_attn(query=q, key=k, value=x2)
        x = x + self.dropout1(x2)
        if query_mask is not None:
            x = x.masked_fill(query_mask.unsqueeze(-1), 0.0)
        # cross attention
        x2 = self.norm2(x)
        x2, cross_attn_map = self.cross_attn(query=self.with_pos_embed(x2, query_pos),
                                             key=self.with_pos_embed(source, source_pos), value=source,
                                             key_padding_mask=key_padding_mask)
        x = x + self.dropout2(x2)
        if query_mask is not None:
            x = x.masked_fill(query_mask.unsqueeze(-1), 0.0)
        # FFNN
        x2 = self.norm3(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout3(x2)
        if query_mask is not None:
            x = x.masked_fill(query_mask.unsqueeze(-1), 0.0)
        return x, cross_attn_map


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")

