from torch.nn import functional as F
import torch
from torch import nn
from attention import MultiHeadSelfAttention, MultiHeadCrossAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads, drop):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.drop = drop

        self.self_att = MultiHeadSelfAttention(d_model, d_k, d_v, num_heads, None)
        self.cross_att = MultiHeadCrossAttention(d_model, d_k, d_v, num_heads, None)

    def forward(self, x, K, V, g):
        x = self.self_att(x)
        x = self.cross_att(x, K, V, g)

        return x


class GlobalAdaptiveDecoder(nn.Module):
    def __init__(self, num_layers, d_model, d_k, d_v, num_heads, drop):
        super(GlobalAdaptiveDecoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.drop = drop

        self.mask = None  # Need to re-evaluate this...

        self.decode_layers = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, num_heads, drop, self.mask) for i in range(self.num_layers)])
        self.ffn = nn.Linear(d_model, d_model)

    def forward(self, x):
        for l in self.decode_layers:
            x = l.forward(x)

        x = torch.softmax(self.ffn(x), -1)

        return x
