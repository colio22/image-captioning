from torch.nn import functional as F
import torch
from torch import nn
import numpy as np
import math


class SelfAttention(nn.Module):
    """
    Class that performs SelfAttention on a given input 
    """

    def __init__(self, d_in, d_k, d_v):
        super(SelfAttention, self).__init__()

        self.d_in = d_in    # Dimensionality of model
        self.d_k = d_k      # Dimension of key/query weights
        self.d_v = d_v      # Dimension of value weights

        # Create fully-connected linear network layers
        self.q_layer = nn.Linear(d_in, d_k)   # Input to Queries
        self.k_layer = nn.Linear(d_in, d_k)   # Input to Keys
        self.v_layer = nn.Linear(d_in, d_v)   # Input to Values

    def attention_score(self, Q, K, V, batch_size, mask=None):
        """
        Computes scaled dot-product attention.
        Parameters:
        Q (torch.Tensor): Query matrix of shape (batch_size, num_heads, seq_len, d_k)
        K (torch.Tensor): Key matrix of shape (batch_size, num_heads, seq_len, d_k)
        V (torch.Tensor): Value matrix of shape (batch_size, num_heads, seq_len, d_v)
        mask (torch.Tensor): Optional attention mask of shape (batch_size, 1, 1, seq_len)
        Returns:
        torch.Tensor: The attention output of shape (batch_size, num_heads, seq_len, d_v)

        Code based on notebook from assignment 4.
        """

        d_k = Q.size(-1)

        if batch_size > 1:  # If batched input
            # Batch multiply Q and K and then scale
            qk = torch.bmm(Q, torch.transpose(K, 1, 2)) / math.sqrt(d_k)
        else:   # If not batched input, standard matrix multiplication
            qk = torch.matmul(Q, torch.transpose(K, 0, 1)) / math.sqrt(d_k)

        if mask != None:  # If mask provided
            # Replace all values masked with False with -infinity
            # Pytorch documentation of masked_fill revealed need to flip mask with logical_n
            qk = qk.masked_fill(mask.logical_not(), -np.inf)

        # Apply softmax to QK matrix
        weights = torch.softmax(qk, -1)

        if batch_size > 1:  # If batched input
            # Multiply with V
            score = torch.bmm(weights, V)
        else:   # If not batched input, standard matrix multiplication
            score = torch.matmul(weights, V)

        return score, weights

    def forward(self, X, batch_size, mask=None):
        """
        Forward pass through self-attention.
            X: Input tensor to attend
            batch_size: Size of batch for X
            mask: Optional attention mask to apply
        """

        # Connect layers in network
        q = self.q_layer(X.float())   # Connect input to Queries
        k = self.k_layer(X.float())   # Connect input to Keys
        v = self.v_layer(X.float())   # Connect input to Values

        # Use self-attention function to attend output
        out, weight = self.attention_score(q, k, v, batch_size, mask)
        return out


class CrossAttention(nn.Module):
    """
    Class that performs cross attention during decoding stage
    """

    def __init__(self, d_in, d_k, d_v):
        super(CrossAttention, self).__init__()

        self.d_in = d_in    # Dimensionality of model
        self.d_k = d_k      # Key/Query matrix dim
        self.d_v = d_v      # Value matrix dim

        # Create fully-connected linear network layers
        self.q_layer = nn.Linear(d_in, d_k)   # Input to Queries
        self.k_layer = nn.Linear(d_in, d_k)   # Input to Keys
        self.v_layer = nn.Linear(d_in, d_v)   # Input to Values

    def attention_score(self, Q, K, V, batch_size, mask=None):
        """
        Computes scaled dot-product attention.
        Parameters:
        Q (torch.Tensor): Query matrix of shape (batch_size, num_heads, seq_len, d_k)
        K (torch.Tensor): Key matrix of shape (batch_size, num_heads, seq_len, d_k)
        V (torch.Tensor): Value matrix of shape (batch_size, num_heads, seq_len, d_v)
        mask (torch.Tensor): Optional attention mask of shape (batch_size, 1, 1, seq_len)
        Returns:
        torch.Tensor: The attention output of shape (batch_size, num_heads, seq_len, d_v)

        Code based on notebook from assignment 4.
        """

        d_k = Q.size(-1)
        if batch_size > 1:  # If batched input
            # Batch multiply Q and K and then scale
            qk = torch.bmm(Q, torch.transpose(K, 1, 2)) / math.sqrt(d_k)
        else:   # If unbatched, standard matrix multiplication
            qk = torch.matmul(Q, torch.transpose(K, 0, 1)) / math.sqrt(d_k)

        # Apply mask if given
        if mask != None:
            qk = qk.masked_fill(mask.logical_not(), -np.inf)

        # Softmax of QK matrix
        weights = torch.softmax(qk, -1)

        if batch_size > 1:  # If batched input
            # Multiply with V
            score = torch.bmm(weights, V)
        else:   # If unbatched, standard matrix multiplication
            score = torch.matmul(weights, V)

        return score, weights

    def forward(self, Q, K, V, batch_size, mask=None):
        """
        Forward pass through cross attention.
        More than one input tensor allowed because Keys, queries,
        and values are provided by encoder output
            Q: Query matrix to use
            K: Key matrix to use
            V: Value matrix to use
            batch_size: Size of batch for Q,K,V
            mask: Optional mask to use during attention
        """

        # Connect layers in network
        q = self.q_layer(Q)   # Connect input to Queries
        k = self.k_layer(K)   # Connect input to Queries
        v = self.v_layer(V)   # Connect input to Queries

        # Use existing self-attention function to attend output
        out, weight = self.attention_score(q, k, v, batch_size, mask)
        return out

    
class MultiHeadSelfAttention(nn.Module):
    """
    Used to perform mutli-headed self attention during encoding and decoding
    """

    def __init__(self, d_in, d_k, d_v, num_heads, mask=None):
        super(MultiHeadSelfAttention, self).__init__()
        # Save input parmeters
        self.d_in = d_in            # Dim of model
        self.d_k = d_k              # Dim of keys/queries
        self.d_v = d_v              # Dim of values
        self.num_heads = num_heads  # Number of attention heads to use

        # Save attention heads as ModuleList for dynamic layer usage
        self.heads = nn.ModuleList([SelfAttention(d_in, int(d_k/num_heads), int(d_v/num_heads)) for i in range(num_heads)])
        # Create fully-connected, linear output layer
        self.out = nn.Linear(d_v, d_v)

    def forward(self, X, batch_size, mask=None):
        """
        Forward pass of input tensor through multi head attention
            X: Input block of features
            batch_size: Size of batch in X
            mask: Optional mask to use in attention
        """

        # Create empty tensor for concatenated attention head outputs
        if batch_size > 1:  # If batched
            outputs = torch.zeros([batch_size, X.size(1), 0], device=X.device)
        else:   # If not batched
            outputs = torch.zeros([X.size(0), 0], device=X.device)

        # Loop through each attention head layer
        for l in self.heads:
          # Concatenate output of layer with previous outputs
          outputs = torch.cat((outputs, l.forward(X, batch_size, mask)), -1)
        # Pass concatenated matrix through fully-connected output layer
        outputs = self.out(outputs)
        return outputs


class MultiHeadCrossAttention(nn.Module):
    """
    Used to perform multi-headed cross attention during decoding
    """
    
    def __init__(self, d_in, d_k, d_v, num_heads, mask=None):
        super(MultiHeadCrossAttention, self).__init__()

        self.d_in = d_in            # Dim of model
        self.d_k = d_k              # Dim of keys/queries
        self.d_v = d_v              # Dim of values
        self.num_heads = num_heads  # Number of attention heads

        # Save attention heads as ModuleList for dynamic layer usage
        self.heads = nn.ModuleList([CrossAttention(d_in, int(d_k/num_heads), int(d_v/num_heads)) for i in range(num_heads)])
        # Create fully-connected, linear output layer
        self.out = nn.Linear(d_v, d_v)

    def forward(self, X, K, V, g, batch_size, mask=None):
        """
        Forward pass of input tensors through cross attention.
            X: Input to be used as query matrix
            K: Key matrix. Comes from encoder output
            V: Value matrix. Comes from encoder output
            g: Global feature from LSTM output
            batch_size: Batch size of input tensors
            mask: Optional mask to use during attention
        """

        # Create empty tensor for concatenated attention head outputs
        if batch_size > 1:  # If input batched
            outputs = torch.zeros([batch_size, X.size(1), 0], device=X.device)
        else:       # If not batched
            outputs = torch.zeros([X.size(0), 0], device=X.device)

        # Loop through each attention head layer
        for l in self.heads:
          att = l.forward(X, K, V, batch_size, mask)
          # Concatenate output of layer with previous outputs
          outputs = torch.cat((outputs, att), -1)

        # Mix in global feature
        outputs = outputs + (X * g)

        # Pass concatenated matrix through fully-connected output layer
        outputs = self.out(outputs)

        return outputs