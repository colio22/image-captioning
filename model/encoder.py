from torch.nn import functional as F
import torch
from torch import nn
from model.attention import MultiHeadSelfAttention
import tensorflow as tf

class GlobalEncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k =64, d_v=64, num_heads=8, drop=0.1):
        super(GlobalEncoderLayer, self).__init__()
        self.d_model=d_model        # Feature dimension
        self.d_k=d_k                # Key and Query attnetion dimension
        self.d_v=d_v                # Value attention dimension
        self.num_heads=num_heads    # Number of attention heads

        # Layers per encoder
        self.att_layer = MultiHeadSelfAttention(d_model, d_k, d_v, num_heads)
        self.ff_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.att_layer(x, mask)       # In: seq_length x d_k. Out: seq_len x d_v
        x = self.ff_layer(x)        # In: seq_len x d_v. Out: seq_len x d_v
        x = self.norm(F.relu(x))    # Normalize and activate

        return x
    
class GlobalEnhancedEncoder(nn.Module):
    def __init__(self, num_layers, d_model, d_k, d_v, num_heads, drop):
        super(GlobalEnhancedEncoder, self).__init__()
        self.num_layers = num_layers    # Number of encoder layers
        self.d_model = d_model          # Feature size
        self.d_k = d_k                  # Query and Key attention dims
        self.d_v = d_v                  # Value attention dims
        self.num_heads = num_heads      # Number of attention heads
        self.drop = drop                # Percent to drop out

        # Encoder Layers
        self.encode_layers = nn.ModuleList([GlobalEncoderLayer(d_model, d_k, d_v, num_heads, drop) for i in range(num_layers)])
        # LSTM Blocks for global memory
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size=d_model*2, hidden_size=d_model) for i in range(num_layers)])
        # Output LSTM block
        self.final_lstm = nn.LSTM(input_size=d_model*2, hidden_size=d_model)

    def forward(self, x, mask=None):
        g = torch.rand([50, 1, self.d_model], device=x.device) # Initialize LSTM with random global feature
        
        # For each encoder layer and LSTM block
        for l, e in zip(self.lstm_layers, self.encode_layers):
            # Isolate global feature to feed into LSTM
            index = torch.ones([50, 1, self.d_model], device=x.device)
            index = len(x)*index
            print(f'=== Input tensor: {x.shape}. Index tensor: {index.shape}')
            g_in = torch.gather(x, 0, index.long())  # Use 0 for the dim argument

            # Concatenate with output of last LSTM layer to feed to next block
            g = torch.cat((g, g_in), -1)

            # Pass global feature to next layer LSTM
            g = l.forward(g.type(torch.float32))
            # Pass total input to next encoder layer
            x = e.forward(x, mask)

        # Final LSTM block
        g = self.final_lstm(g)
        return x, g
    
            