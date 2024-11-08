from torch.nn import functional as F
import torch
from torch import nn
from model.encoder import GlobalEnhancedEncoder
from model.decoder import GlobalAdaptiveDecoder


class GlobalEnhancedTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, d_k, d_v, num_heads, num_layers, drop):
        super(GlobalEnhancedTransformer, self).__init__()
        self.vocab_size = vocab_size    # Number of words in vocabulary
        self.max_len = max_len          # Max length of word sequence
        self.d_model = d_model          # Dimension of model features
        self.d_k = d_k                  # Demensions of keys and queries
        self.d_v = d_v                  # Dimensions of attention values
        self.num_heads = num_heads      # Number of attention heads per layer
        self.drop = drop                # Percentage to drop in fully connected layers
        self.num_layers = num_layers    # Number of encoder/decoder layers

        # Encoder Block
        self.encoder = GlobalEnhancedEncoder(num_layers, d_model, d_k, d_v, num_heads, drop)
        # Decoder Block
        self.decoder = GlobalAdaptiveDecoder(vocab_size, max_len, 0, num_layers, d_model, d_k, d_v, num_heads, drop)

    def forward(self, img, seq):
        # Create global feature composite of all other features
        x = torch.sum(img, dim=1)
        # Normalize by the number of non-padding feature vectors
        num_padding = len(x) - len(torch.nonzero(x))
        g = x / (len(x) - num_padding)

        # Add global feature to list of features
        v = torch.cat((img, g.unsqueeze(1)), -2)

        # Proceed with transformer
        img_out, g_out =  self.encoder(v)
        seq_out = self.decoder(seq, img_out, img_out, g_out)

        return seq_out
        