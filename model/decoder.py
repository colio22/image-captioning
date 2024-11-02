from torch.nn import functional as F
import torch
from torch import nn
from attention import MultiHeadSelfAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads, drop):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.drop = drop

        self.self_att = MultiHeadSelfAttention(d_model, d_k, d_v, num_heads, None)
        self.cross_att = MultiHeadSelfAttention(d_model, d_k, d_v, num_heads, None)
        self.ffn = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.self_att(x)
