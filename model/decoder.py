from torch.nn import functional as F
import torch
from torch import nn
from model.attention import MultiHeadSelfAttention, MultiHeadCrossAttention
import math


class DecoderLayer(nn.Module):
    """
    Single layer to be used in decoder
    """

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

    def forward(self, x, K, V, g, batch_size, mask=None):
        """
        Forward pass of text sequence and encoder outputs through decoder layer
            x: Input sequence to use as attention query
            K: Encoder output to use as attention keys
            V: Encoder output to use as attention values
            g: Global feature output from LSTM
            batch_size: Batch size of all input tensors
        """

        # Apply self attention
        x = self.self_att(x, batch_size, mask)           # In: seq_len x d_k. Out: seq_len x d_v
        # Apply cross attention
        x = self.cross_att(x, K, V, g, batch_size) # In: seq_len x d_k. Out: seq_len x d_v

        return x


class GlobalAdaptiveDecoder(nn.Module):
    """
    Decoder block of the Global Adaptive Decoder
    """

    def __init__(self, vocab_size, padding, num_layers, d_model, d_k, d_v, num_heads, drop):
        super(GlobalAdaptiveDecoder, self).__init__()
        self.vocab_size = vocab_size    # Number of words in vocabulary
        self.padding = padding          # Padding used for embedding words
        self.num_layers = num_layers    # Number of decode layers
        self.d_model = d_model          # Dimensionality of decoder
        self.d_k = d_k                  # Dimension of keys and quries
        self.d_v = d_v                  # Dimension of attention values
        self.num_heads = num_heads      # Number of heads used in attention
        self.drop = drop                # Dropout percentage

        # List of decoder layers
        self.decode_layers = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, num_heads, drop) for i in range(self.num_layers)])
        # Feed-forward network for output of each decoder layer
        self.ffn = nn.Linear(d_model, vocab_size)   # Output as 1-hot encoding for entire vocabulary
        # Embedding layer for input sequence
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding)

    def get_positional_encoding(self, d_model, seq_len, device):
        """
        Generate a matrix where each row corresponds to the sinusoidal positional encoding
            d_model: Model dimensionality
            seq_len: Length of given text sequence
            device: GPU device to use

        Code re-used from assignment 4
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
        """
        Creates a mask to use in attention that prevents items
        from 'seeing the future.'
            seq_len: Length of input sequence
            device: GPU device to use

        Code re-used from assignment 4
        """
        
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

    def forward(self, x, K, V, g, batch_size):
        """
        Forward pass of input text sequence and encoder output through entire decoder.
            x: Input text sequence, tokenized
            K: Output of encoder
            V: Output of encoder
            g: global feature, output of LSTM network
            batch_size: Batch size of all input tensors
        """
        
        # Embed and encode word sequence
        out = self.word_emb(x) + self.get_positional_encoding(self.d_model, x.size(1), x.device)

        # If inputs not batched, remove batch dimension
        if batch_size <= 1:
           out = torch.squeeze(out, dim=0)

        # Create a mask to prevent things from the future
        mask = self.create_mask(x.size(1), x.device)

        # Pass sequence and encoder results through each decode layer
        for l in self.decode_layers:
            out = l.forward(out, K, V, g, batch_size, mask)

        # Activate output with fully connected layer
        out = torch.softmax(self.ffn(out), -1)

        return out
