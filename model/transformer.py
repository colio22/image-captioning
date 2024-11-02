from torch.nn import functional as F
import torch
from torch import nn
from model.encoder import GlobalEnhancedEncoder
from model.decoder import GlobalAdaptiveDecoder


class GlobalEnhancedTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, d_k, d_v, num_heads, num_layers, drop):
        super(GlobalEnhancedTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.drop = drop
        self.num_layers = num_layers

        self.encoder = GlobalEnhancedEncoder(num_layers, d_model, d_k, d_v, num_heads, drop)
        self.decoder = GlobalAdaptiveDecoder(vocab_size, max_len, 0, num_layers, d_model, d_k, d_v, num_heads, drop)

    def forward(self, img, seq):
        img_out, g =  self.encoder(img)
        seq_out = self.decoder(seq, img_out, img_out, g)

        return seq_out
        