import torch

class Attention2(torch.nn.Module):

    """
    Same as Attention but using more utilities form pytorch
    """
   
    def __init__(self, token_vector_dim, weight_vector_dim, qkv_bias=False):
        super().__init__()
        self.query_weights = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
        self.key_weights = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
        self.value_weights = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
       
    
    def forward(self, embeddings):
        keys = self.key_weights(embeddings)
        queries = self.query_weights(embeddings)
        values = self.value_weights(embeddings)

        scores = queries @ keys.T
        weights = torch.softmax(scores / keys.shape[-1]**0.5, dim=-1)
        return weights @ values

