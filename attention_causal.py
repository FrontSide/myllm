import torch

class AttentionCausal(torch.nn.Module):

    """
    Attention Mechanism with causal attention (page 75)
    """
   
    def __init__(self, token_vector_dim, weight_vector_dim, qkv_bias=False):
        super().__init__()
        self.query_weights = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
        self.key_weights = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
        self.value_weights = torch.nn.Linear(token_vector_dim, weight_vector_dim, bias=qkv_bias)
       
   
    def weights(self, embeddings):
        keys = self.key_weights(embeddings)
        queries = self.query_weights(embeddings)
        scores = queries @ keys.T
        return torch.softmax(scores / keys.shape[-1]**0.5, dim=-1)

    def simple_masked_weights(self, embeddings):
        """
        Applies a mask so that attention weights are only calculated for 
        words prior to and including the word itself.
        We are not attentding "forward".
        To do this we multiply the weights with a matrix of 1s and 0s
        and normalise the rows again so the sum of all values is 0. (see page 75/76)
        """
        weights = self.weights(embeddings)
        context_length = weights.shape[0]
        mask = torch.tril(torch.ones(context_length, context_length))
        masked_simple = weights * mask
        row_sums = masked_simple.sum(dim=-1, keepdim=True)
        return masked_simple / row_sums


    def forward(self, embeddings):
        values = self.value_weights(embeddings)
        return self.weights(embeddings) @ values

