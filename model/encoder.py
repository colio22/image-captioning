from torch.nn import functional as F
import torch
from torch import nn
from model.attention import MultiHeadSelfAttention
import TensorFlow as tf

class GlobalEncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k =64, d_v=64, num_heads=8, drop=0.1, mask=None):
        super(GlobalEncoderLayer, self).__init__()

        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.num_heads=num_heads
        self.d_ff=d_ff

        self.att_layer = MultiHeadSelfAttention(d_model, d_k, d_v, num_heads, mask)
        self.ff_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.att_layer(x)
        x = self.ff_layer(x)
        x = self.norm(F.relu(x))

        return x
    
class GlobalEnhancedEncoder(nn.Module):
    def __init__(self, num_layers, d_model, d_k, d_v, num_heads, drop):
        super(GlobalEnhancedEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.drop = drop

        self.mask = None   # Not sure if necessary or not...

        self.encode_layers = nn.ModuleList([GlobalEncoderLayer(d_model, d_k, d_v, num_heads, drop, self.mask) for i in range(num_layers)])
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size=d_model*2, hidden_size=d_model) for i in range(num_layers)])
        self.final_lstm = nn.LSTM(input_size=d_model*2, hidden_size=d_model)

    def forward(self, x):
        intermediates = []
        g = torch.rand(1, self.d_model)
        for l, e in zip(self.lstm_layers, self.encode_layers):
            g_in = tf.gather(x, index=[len(x)-1], dim=0)
            g = torch.cat((g, g_in) -1)
            g = l.forward(g)
            x = e.forward(x)

        g = self.final_lstm(g)
        return x, g
    
            