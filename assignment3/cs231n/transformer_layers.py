import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        """
        [[0],
        [1],) * ([0.0000, -0.3333, -0.6667])
        ======== 
        [[ 0.0000, -0.0000, -0.0000],
        [ 0.0000, -0.3333, -0.6667]])
        """
        len_idx = torch.arange(0, max_len).unsqueeze(1) 
        dim_idx = torch.arange(0, embed_dim, step=2)    # for example: embed_dim=8, dim_idx=[0,2,4,6]
        # (L, 1) * (1, D/2)(broadcast to l, D/2) => (L, D/2)
        idx_matrix = len_idx * 1e4 ** (-dim_idx / embed_dim)  # [0.0, 0.25, 0.5, 0.75] -> [1.0, 0.31622777, 0.1, 0.03162278] -> [0, ... ,7](L,1) *** (1,D/2) => (L,D/2)
        pe[:, :, 0::2] = torch.sin(idx_matrix) # 偶数列，0::2 is short for 0:len:2
        pe[:, :, 1::2] = torch.cos(idx_matrix) # 奇数列，1::2 is short for 1:len:2
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        
        output = x + self.pe[:,:S,:] 
        output = self.dropout(output)


        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        H = self.n_head
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        """
        N is the batch size, S is the source sequence length, T is the target sequence length, 
        # and E is the embedding dimension
        """
        Q = self.query(query).reshape(N, S, H, E // H).permute(0, 2, 1, 3) # (N,S,E) -> (N,S,H,E//H) -> (N,H, S,    E//H)
        K = self.key(key).reshape(N, T, H, E // H).permute(0, 2, 3, 1) # (N,T,E) -> (N,T,H,E//H) ->     (N,H, E//H, T)
        V = self.value(value).reshape(N, T, H, E // H).permute(0, 2, 1, 3) # (N,T,E) -> (N,T,H,E//H) -> (N,H, T,    E//H)
        QK = Q.matmul(K) / torch.sqrt(torch.Tensor([E / H])) # (N, H, S, E//H) * (N, H, E//H, T) -> (N, H, S, T)
        if attn_mask is not None:
            QK = QK.masked_fill(attn_mask == 0, -1e9)
        QKV = self.attn_drop(F.softmax(QK, dim=-1)).matmul(V) # (N, H, S, T) * (N, H, T, E//H) -> (N, H, S, E//H) 
        output = self.proj(QKV.permute(0, 2, 1, 3).reshape(N, S, E)) # (N, H, S, E//H) -> (N, S, H, E//H) -> (N, S, E)
        return output


