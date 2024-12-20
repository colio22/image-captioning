from torch.nn import functional as F
import torch
from torch import nn
from model.attention import MultiHeadSelfAttention
import tensorflow as tf


class GlobalEncoderLayer(nn.Module):
    """
    Single layer of the encoder block
    """

    def __init__(self, d_model=512, d_k=64, d_v=512, num_heads=8, drop=0.1):
        super(GlobalEncoderLayer, self).__init__()
        self.d_model=d_model        # Feature dimension
        self.d_k=d_k                # Key and Query attnetion dimension
        self.d_v=d_v                # Value attention dimension
        self.num_heads=num_heads    # Number of attention heads

        # Layers per encoder
        self.att_layer = MultiHeadSelfAttention(d_model, d_k, d_v, num_heads)
        self.ff_layer = nn.Linear(d_v, d_v) # Feed-forward network (no dim reductions)
        self.dropout = nn.Dropout(drop)     # Dropout layer to make model lighter
        self.norm = nn.LayerNorm(d_v)       # Normalization layer

    def forward(self, x, batch_size, mask=None):
        """
        Forward pass of image features through layer
            x: sequence of input features
            batch_size: Batch size of x
            mask: Optional mask to use in attention
        """

        x = self.att_layer(x, batch_size, mask)       # In: seq_length x d_k. Out: seq_len x d_v
        x = self.ff_layer(x)        # In: seq_len x d_v. Out: seq_len x d_v
        x = self.norm(F.relu(x))    # Normalize and activate

        return x
    

class GlobalEnhancedEncoder(nn.Module):
    """
    Entire encoder block of Global Enhanced Transformer
    """
    
    def __init__(self, num_layers, feature_size, d_model, d_k, d_v, num_heads, drop):
        super(GlobalEnhancedEncoder, self).__init__()
        self.num_layers = num_layers    # Number of encoder layers
        self.feature_size = feature_size # Dimension of input features
        self.d_model = d_model          # Feature size
        self.d_k = d_k                  # Query and Key attention dims
        self.d_v = d_v                  # Value attention dims
        self.num_heads = num_heads      # Number of attention heads
        self.drop = drop                # Percent to drop out

        # Encoder Layers
        self.initial_encode = GlobalEncoderLayer(feature_size, d_k, d_model, num_heads, drop)
        self.encode_layers = nn.ModuleList([GlobalEncoderLayer(d_model, d_k, d_v, num_heads, drop) for i in range(num_layers-1)])
        # LSTM Blocks for global memory
        self.initial_lstm = nn.LSTM(input_size=feature_size*2, hidden_size=d_model, num_layers=1, batch_first=True)
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size=d_model*2, hidden_size=d_model, num_layers=1, batch_first=True) for i in range(num_layers-1)])
        self.final_lstm = nn.LSTM(input_size=d_model*2, hidden_size=d_model, num_layers=1, batch_first=True)

    def forward(self, x, batch_size, mask=None):
        """
        Forward pass of input features through encoder
        """

        # If batched input
        if batch_size > 1:
            g = torch.rand([batch_size, 1, self.feature_size], device=x.device) # Initialize LSTM with random global feature
            h = torch.zeros([1, batch_size, self.d_model], device=x.device)
            c = torch.zeros([1, batch_size, self.d_model], device=x.device)

            # Isolate global feature to feed into LSTM
            index_0 = torch.ones([batch_size, 1, self.feature_size], device=x.device)
            index_0 = (len(x)-1)*index_0
            index = torch.ones([batch_size, 1, self.d_model], device=x.device)
            index = (len(x)-1)*index
        else:   # If unbatched input
            g = torch.rand([1, self.feature_size], device=x.device) # Initialize LSTM with random global feature
            h = torch.zeros([1, self.d_model], device=x.device)
            c = torch.zeros([1, self.d_model], device=x.device)

            # Isolate global feature to feed into LSTM
            index_0 = torch.ones([1, self.feature_size], device=x.device)
            index_0 = (len(x)-1)*index_0
            index = torch.ones([1, self.d_model], device=x.device)
            index = (len(x)-1)*index
        
        #Initial encoder and LSTM
        g_in = torch.gather(x, 0, index_0.long())  # Use 0 for the dim argument

        # Concatenate with output of last LSTM layer to feed to next block
        g = torch.cat((g, g_in), -1)

        # Pass global feature to next layer LSTM
        g, (h, c) = self.initial_lstm(g.type(torch.float32), (h, c))
        # Pass total input to next encoder layer
        x = self.initial_encode(x, batch_size, mask)
        
        # For each encoder layer and LSTM block
        for l, e in zip(self.lstm_layers, self.encode_layers):
            g_in = torch.gather(x, 0, index.long())  # Use 0 for the dim argument

            # Concatenate with output of last LSTM layer to feed to next block
            g = torch.cat((g, g_in), -1)

            # Pass global feature to next layer LSTM
            g, (h, c) = l.forward(g.type(torch.float32), (h, c))

            # Pass total input to next encoder layer
            x = e.forward(x, batch_size, mask)


        # Final LSTM block
        g_in = torch.gather(x, 0, index.long())  # Use 0 for the dim argument
        # Concatenate with output of last LSTM layer to feed to next block
        g = torch.cat((g, g_in), -1)
        g, (h, c) = self.final_lstm(g, (h,c))

        return x, g
    
            