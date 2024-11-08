from torch.nn import functional as F
import torch
from torch import nn
from model.attention import MultiHeadSelfAttention, MultiHeadCrossAttention
from common.models.transformer import sinusoid_encoding_table
import math


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
        x = self.cross_att(x, K, V, g, mask) # In: seq_len x d_k. Out: seq_len x d_v

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
        # self.pos_enc = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len+1, d_model, 0), freeze=True)

    def get_positional_encoding(self, d_model, seq_len, device):
        """
        Generate a matrix where each row corresponds to the positional encoding for a 
        """

        # Initialize empty matrix to hold positional encodings
        pe = torch.zeros([seq_len, d_model], device=device)
        # Iterate over entire encoding matrix
        for i in range(0, seq_len):  # Loop over each row
          for j in range(0, int(d_model/2)): # Loop over half of dimensions
            # Apply sin encoding to 2*dimension index of matrix
            pe[i][2*j] = math.sin((i)/pow(10000, (2*j)/d_model))
            # Apply cos encoding to (2*dimension)+1 index of matrix
            pe[i][(2*j)+1] = math.cos((i)/pow(10000, (2*j)/d_model))
        return pe

    def create_mask(self, seq_len, device):
        # Create empty bool mask
        mask = torch.zeros([seq_len, seq_len], device=device, dtype=torch.bool)
        # Loop over entire mask
        for i in range(0, seq_len):
          for j in range(0, seq_len):
            if j > i:   # If in upper triangle, set to false to mark for masking out
              mask[i][j] = False
            else:       # Else set to true to mark for keeping
              mask[i][j] = True
        return mask

    def forward(self, x, K, V, g):
        # b_s, seq_len = input.shape[:2]
        # seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)

        print(f'===Shape of x for pos enc: {x.shape}')
        out = self.word_emb(x) + self.get_positional_encoding(self.d_model, x.size(1), x.device)
        mask = self.create_mask(x.size(1), x.device)
        for l in self.decode_layers:
            out = l.forward(out, K, V, g, mask)

        out = torch.softmax(self.ffn(out), -1)

        return out
