from torch.nn import functional as F
import torch
from torch import nn
from model.encoder import GlobalEnhancedEncoder
from model.decoder import GlobalAdaptiveDecoder

class PositionalEncoding:
    def __init__(self, d_model, max_len):
        self.d_model = d_model
        self.max_len = max_len
        self.positional_encoding = self._get_positional_encoding()

    def _get_positional_encoding(self):
        """
        Generate a matrix where each row corresponds to the positional encoding for a 
        """
        # Initialize empty matrix to hold positional encodings
        pe = torch.zeros(self.max_len, self.d_model)
        # Iterate over entire encoding matrix
        for i in range(0, self.max_len):  # Loop over each row
          for j in range(0, int(self.d_model/2)): # Loop over half of dimensions
            # Apply sin encoding to 2*dimension index of matrix
            pe[i][2*j] = math.sin((i)/pow(10000, (2*j)/self.d_model))
            # Apply cos encoding to (2*dimension)+1 index of matrix
            pe[i][(2*j)+1] = math.cos((i)/pow(10000, (2*j)/self.d_model))
        return pe

    def get_encoding(self):
        """
        Returns the positional encoding matrix.
        """
        return self.positional_encoding

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, d_k, d_v, num_heads, num_layers, drop):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.drop = drop
        self.num_layers = num_layers

        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)