from torch.nn import functional as F
import torch
from torch import nn
from model.attention import MultiHeadSelfAttention, MultiHeadCrossAttention
from common.models.transformer import sinusoid_encoding_table

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads, drop):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model      # Size of features
        self.d_k = d_k              # Query and Key attention dims
        self.d_v = d_v              # Value attention dim
        self.num_heads = num_heads  # Number of attention heads
        self.drop = drop            # Dropout percentage

        # Layer for standard multi head attention
        self.self_att = MultiHeadSelfAttention(d_model, d_k, d_v, num_heads, None)
        # Layer for global cross attention
        self.cross_att = MultiHeadCrossAttention(d_model, d_k, d_v, num_heads, None)

    def forward(self, x, K, V, g, mask=None):
        x = self.self_att(x, mask)           # In: seq_len x d_k. Out: seq_len x d_v
        x = self.cross_att(x, K, V, g, mask)

        return x


class GlobalAdaptiveDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, padding, num_layers, d_model, d_k, d_v, num_heads, drop):
        super(GlobalAdaptiveDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.padding = padding
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.drop = drop

        self.decode_layers = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, num_heads, drop) for i in range(self.num_layers)])
        self.ffn = nn.Linear(d_model, d_model)
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding)
        self.pos_enc = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len+1, d_model, 0), freeze=True)

    def forward(self, x, K, V, g, mask=None):
        b_s, seq_len = input.shape[:2]
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)

        out = self.word_emb(x) + self.pos_enc(seq)
        for l in self.decode_layers:
            x = l.forward(x)

        x = torch.softmax(self.ffn(x), -1)

        return x
