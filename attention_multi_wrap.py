import torch
from attention_causal2 import AttentionCausal2

class MultiHeadAttentionWrapper(torch.nn.Module):

    def __init__(self, token_vector_dim, weight_vector_dim, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
                [AttentionCausal2(
                    token_vector_dim, weight_vector_dim, context_length, dropout, qkv_bias
                    ) for _ in range(num_heads)]
                )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


