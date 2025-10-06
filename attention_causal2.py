import torch

class AttentionCausal2(torch.nn.Module):

    """
    Same as AttentionCausal but more compact  
    """
   
    def __init__(self, token_vector_dim, weight_vector_dim, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.query_weights = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
        self.key_weights = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
        self.value_weights = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias) 
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

    def forward(self, embeddings):
        num_batches, num_tokens, token_vector_dim = embeddings.shape
        keys = self.key_weights(embeddings)
        queries = self.query_weights(embeddings)
        values = self.value_weights(embeddings)

        scores = queries @ keys.transpose(1, 2)
        scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        weights = torch.softmax(
            scores / keys.shape[-1]**0.5, dim=-1
        )
        weights = self.dropout(weights)

        return weights @ values

