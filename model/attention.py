from torch.nn import functional as F
import torch
from torch import nn
import numpy as np
import math

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_k, d_v):
        super(SelfAttention, self).__init__()

        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v

        # Create fully-connected linear network layers
        self.q_layer = nn.Linear(d_in, d_k)   # Input to Queries
        self.k_layer = nn.Linear(d_in, d_k)   # Input to Keys
        self.v_layer = nn.Linear(d_in, d_v)   # Input to Values

    def att_score(Q, K, V, mask=None):
        d_k = Q.size(-1)
        # Batch multiply Q and K and then scale
        qk = torch.bmm(Q, torch.transpose(K, 1, 2)) / math.sqrt(d_k)

        if mask != None:
            qk = qk.masked_fill(mask.logical_not(), -np.inf)

        # Softmax of QK matrix
        weights = torch.softmax(qk, -1)
        # Multiply with V
        score = torch.bmm(weights, V)
        return score, weights

    def attention_score(self, Q, K, V, mask=None):
        """
        Computes scaled dot-product attention.
        Parameters:
        Q (torch.Tensor): Query matrix of shape (batch_size, num_heads, seq_len, d_k)
        K (torch.Tensor): Key matrix of shape (batch_size, num_heads, seq_len, d_k)
        V (torch.Tensor): Value matrix of shape (batch_size, num_heads, seq_len, d_v)
        mask (torch.Tensor): Optional attention mask of shape (batch_size, 1, 1, seq_len)
        Returns:
        torch.Tensor: The attention output of shape (batch_size, num_heads, seq_len, d_v)
        """

        d_k = Q.size(-1)

        # Batch multiply Q and K and then scale
        qk = torch.bmm(Q, torch.transpose(K, 1, 2)) / math.sqrt(d_k)

        if mask != None:  # If mask provided
            # Replace all values masked with False with -infinity
            # Pytorch documentation of masked_fill revealed need to flip mask with logical_n
            print(f'!!!!! Mask shape: {mask.shape}. Q*K resulting shape: {qk.shape}')
            qk = qk.masked_fill(mask.logical_not(), -np.inf)

        # Apply softmax to QK matrix
        weights = torch.softmax(qk, -1)
        # Multiply with V
        score = torch.bmm(weights, V)
        return score, weights

    def forward(self, X, mask=None):
        # Connect layers in network
        q = self.q_layer(X.float())   # Connect input to Queries
        k = self.k_layer(X.float())   # Connect input to Keys
        v = self.v_layer(X.float())   # Connect input to Values

        # Use existing self-attention function to attend output
        out, weight = self.attention_score(q, k, v, mask)
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_in, d_k, d_v):
        super(CrossAttention, self).__init__()

        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v

        # Create fully-connected linear network layers
        self.q_layer = nn.Linear(d_in, d_k)   # Input to Queries
        self.k_layer = nn.Linear(d_in, d_k)   # Input to Keys
        self.v_layer = nn.Linear(d_in, d_v)   # Input to Values

    def attention_score(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        # Batch multiply Q and K and then scale
        print(f'~~~ In att score. Q={Q.shape}. K={K.shape}. V={V.shape}')
        qk = torch.bmm(Q, torch.transpose(K, 1, 2)) / math.sqrt(d_k)

        if mask != None:
            qk = qk.masked_fill(mask.logical_not(), -np.inf)

        # Softmax of QK matrix
        weights = torch.softmax(qk, -1)
        # Multiply with V
        score = torch.bmm(weights, V)
        return score, weights

    def forward(self, Q, K, V, mask=None):
        # Connect layers in network
        q = self.q_layer(Q)   # Connect input to Queries
        k = self.q_layer(K)   # Connect input to Queries
        v = self.q_layer(V)   # Connect input to Queries

        # Use existing self-attention function to attend output
        out, weight = self.attention_score(q, k, v, mask)
        return out

    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_in, d_k, d_v, num_heads, mask=None):
        super(MultiHeadSelfAttention, self).__init__()
        # Save input parmeters
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        # Save attention heads as ModuleList for dynamic layer usage
        self.heads = nn.ModuleList([SelfAttention(d_in, int(d_k/num_heads), int(d_v/num_heads)) for i in range(num_heads)])
        # Create fully-connected, linear output layer
        self.out = nn.Linear(d_v, d_v)

    def forward(self, X, mask=None):
        # Create empty tensor for concatenated attention head outputs
        outputs = torch.zeros([50, X.size(1), 0], device=X.device)
        # Loop through each attention head layer
        for l in self.heads:
          # Concatenate output of layer with previous outputs
          outputs = torch.cat((outputs, l.forward(X, mask)), -1)
        # Pass concatenated matrix through fully-connected output layer
        outputs = self.out(outputs)
        return outputs

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_in, d_k, d_v, num_heads, mask=None):
        super(MultiHeadCrossAttention, self).__init__()
        # Save input parmeters
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        # Save attention heads as ModuleList for dynamic layer usage
        self.heads = nn.ModuleList([CrossAttention(d_in, int(d_k/num_heads), int(d_v/num_heads)) for i in range(num_heads)])
        # Create fully-connected, linear output layer
        self.out = nn.Linear(d_v, d_v)

    def forward(self, X, K, V, g, mask=None):
        # Create empty tensor for concatenated attention head outputs
        outputs = torch.zeros(32, 10, 0)
        # Loop through each attention head layer
        for l in self.heads:
          att = l.forward(X, K, V, mask)
          # Concatenate output of layer with previous outputs
          outputs = torch.cat((outputs, att), -1)

        outputs = outputs + (X * g)
        # Pass concatenated matrix through fully-connected output layer
        outputs = self.out(outputs)
        return outputs