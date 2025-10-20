import torch
from attention_causal2 import AttentionCausal2

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, token_vector_dim, weight_vector_dim, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (weight_vector_dim % num_heads == 0), "weight_vector_dim must be divisible by num_heads"

        self.weight_vector_dim = weight_vector_dim
        self.num_heads = num_heads
        self.head_dim = self.weight_vector_dim // num_heads
        self.weights_query = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
        self.weights_key = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
        self.weights_value = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(weight_vector_dim, weight_vector_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, embeddings):

        b, num_tokens, token_vector_dim = embeddings.shape
        keys = self.weights_key(embeddings)
        queries = self.weights_query(embeddings)
        values = self.weights_value(embeddings)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.weight_vector_dim)

        context_vec = self.out_proj(context_vec)
        return context_vec
