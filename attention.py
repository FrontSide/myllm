import torch

class Attention():

    """
    The trainable attention mechanism for the LLM. 
    This self-attention mechanism is alsop called the "scaled dot-product attention"

    For any given sequence of input tokens, the goal is to calculate attention scores 
    for each token relative to all other tokens. 

    1. The class is initialised with query, key and value (qkv) weights.
    2. For each input sequence, qkv matrices are calculated, whereas each row is a token's (qkv) vector
    3. The attention score for a token (t) relative to another "attained-to" token (a)
        is calculated as the dot product of t's key vector (the relevant row) and a's query vector.
    4. Calculate the attention weights, which are simply the scaled and normalized attention scores.

    """

    def __init__(self, token_vector_dim, weight_vector_dim):
        """
        This implementation of the attention meachanism uses trainable weight matrices.
        For this we need to create 3 weight matrices, named: query, key and value.
        These matrices will be used in calculating the attention weights and,
        thus, the context vectors

        Each of these matrices will be of size token_vector_dim * weight_vector_dim
        This way we can perform matrix multiplications between the token embeddings 
        and the qkv weights to produce embedding-specific qkv vectors.
        These can then be used to calculate the attention score. 

        parameters:
            token_vector_dim:
                The dimension of the embedding vector representing a token 
            weight_vector_dim:
                The dimension of the vector representing a weigth
        """
        torch.manual_seed(123)
        self.weights_vector_dim = weight_vector_dim
        self.query_weights = torch.nn.Parameter(torch.rand(token_vector_dim, weight_vector_dim), requires_grad=False)
        self.key_weights = torch.nn.Parameter(torch.rand(token_vector_dim, weight_vector_dim), requires_grad=False)
        self.value_weights = torch.nn.Parameter(torch.rand(token_vector_dim, weight_vector_dim), requires_grad=False)
       
    
    def query_matrix(self, embeddings):
        """
        Generates the query matrix for the given token embeddings.
        The query matrix is calculated as the token embeddings matrix multiplied 
        by the query weights matrix.
        Each row in the output matrix represents one token's query vector.

        params:
            embeddings must be a tensor of order 2,
                with each row being a vector of embeddings 
                of dimension self.token_vector_dim representing one token.
        """
        return embeddings @ self.query_weights


    def key_matrix(self, embeddings):
        """
        Same as query_matrix but calculates the key matrix
        """
        return embeddings @ self.key_weights 


    def value_matrix(self, embeddings):
        """
        Same as query_matrix but calculates the value matrix
        """
        return embeddings @ self.value_weights


    def scores_for_token(self, embeddings, tok_idx):
        """
        Calculate the attention scores of all tokens in embeddings 
        relative to the "attended" token at index tok_idx 

        The attention scores are the dot product of the query vector
        of the attended to token (tok_idx) and the key vector of a given token.
        """
        query_matrix = self.query_matrix(embeddings)
        key_matrix = self.key_matrix(embeddings)

        return query_matrix[tok_idx] @ key_matrix.T

    def weights_for_token(self, embeddings, tok_idx):
        """
        Calcualte the attention weights for all tokens in embeddings
        relative to the "attended" token at index tok_idx

        The attention weights are the scaled and normalized attention scores, 
        so that the sum of all weights for an input sequence = 1

        The normalization here requires us to scale the scores 
        by divide the scores by the square root of the number of dimensions of the weight vectors. 
        In the case of LLMs which usually use a high number of vector dimensions (>1000),
        we would run in to "small gradients" that lower the training speed.

        This is why this self-attention mechanism is also called the "scaled-dot product attention"
        """
        return torch.softmax(self.scores_for_token(embeddings, tok_idx) / self.weights_vector_dim**0.5, dim=-1)


    def context_vector_for_token(self, embeddings, tok_idx):
        """
        Calculates the context vector for the token at given tok_idx. 

        The context vector is the sum of all input tokens' embeddings
        multiplied by their attention weights.
        The context vector is of the same dimension as the embedding vectors

        For example:
            Say we have the following embeddings (4 tokens, with an embedding vector of 3 dimensions)
            and want to calculate the context vector for tok_idx=1

            embeddings = [
                [0.4, 0.1, 0.8], // tok_idx=0
                [0.5, 0.8, 0.6], // tok_idx=1
                [0.5, 0.8, 0.6], 
                [0.0, 0.8, 0.5]
            ]

            We now calculate the attention_weights for token at index 1, which results in:

            attention_weights = [0.1, 0.2, 0.2, 0.1]

            To calculate the context_vector we now multiply each input/embedding vector by the associated weight 

            [
                [0.4, 0.1, 0.8] * 0.1,
                [0.2, 0.8, 0.6] * 0.2,
                ...etc...
            ]

            We now sum up those vectors 

            context_vector = [0.4, 0.6, 0.5]
        """
        attention_weights = self.weights_for_token(embeddings, tok_idx)
        context_vector = torch.zeros(embeddings[tok_idx].shape)
        for i, x_i in enumerate(embeddings):
            context_vector += attention_weights[i] * x_i
        return context_vector

    def scores(self, embeddings):
        """
        Calculates the attentions scores for all tokens.

        Same as scores_for_token but for all tokens instead of just the one at a given index

        The resulting tensor is of size nr_of_tokens * nr_of_tokens (i.e. N^2),
        since we get a score for each token relative to every other token.

        We could use a nested for look and calculate the dot products one by one or even 
        call the attention_scores_for_token() method, but for performance reason we instead 
        use matrix multiplication which does exactly the same thing.
        """
        return embeddings @ embeddings.T
   
    def weights(self, embeddings):
        """
        Calculates the attention weithts for all tokens.

        Same as weights_for_token but for all tokens.
        """
        return torch.softmax(self.scores(embeddings), dim=-1)

    def context_vectors(self, embeddings):
        """
        Caclulates the context vectors for all tokens.

        Same as context_vector_for_token but for all tokens.
        """
        return self.weights(embeddings) @ embeddings
