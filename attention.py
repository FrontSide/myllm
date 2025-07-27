import torch

class Attention(torch.nn.Module):

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

    The terms query, key and value are borrowed from the database/information retrieval domain.
    <query> is effectively the search term which we use to match against keys.
    For each <key> vector (representing a token) we see how relevant it is for the given attended to token,
    for which we use the query vector i.e. the attention weight. 
    Once the attention weight for each key in respect to the query has been found we apply it to the 
    actual <value> of the given token and compute the context vector.

    Assumption: We don't want to use juse want single vector for all three purposes (query, key, value)
    as we need to be be able to fine-tune all three of them appropriately during training.

    For production use we inherit from torch.nn.Module which gives us some functionalities built-in.
    Technically it's not a requirement.
    It allows us e.g. to simply use this class like 
        context_matrix = Attention(dim, dim)(embeddings)
    """

    def __init__(self, token_vector_dim, weight_vector_dim, requires_grad=False):
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
            requires_grad:
                Should be true for production use to make the weights trainable.
                Note that we will not set a manual seed if this is set to true
        """
        if not requires_grad:
            # No manual see for production use
            torch.manual_seed(123)
        super().__init__()
        self.weights_vector_dim = weight_vector_dim
        self.query_weights = torch.nn.Parameter(torch.rand(token_vector_dim, weight_vector_dim), requires_grad=requires_grad)
        self.key_weights = torch.nn.Parameter(torch.rand(token_vector_dim, weight_vector_dim), requires_grad=requires_grad)
        self.value_weights = torch.nn.Parameter(torch.rand(token_vector_dim, weight_vector_dim), requires_grad=requires_grad)
       
    
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
        Calculate the context vector for the given embeddings 
        relative to the "attended" token at index tok_idx

        The context vector is calculated as the sum of the products 
        of each token's attention weight with its value vector. 
        The attention weight is thus used as a weighing factor that weighs
        the importance of each token's value vector.
        """
        
        return self.weights_for_token(embeddings, tok_idx) @ self.value_matrix(embeddings)

    def forward(self, embeddings):
        """
        Calculates the context vectors for all tokens.
        This is effectively a combination of all the above methods 
        generalised to the full embeddings (vs just one attended to token)

        The above methods were implemented only for learning purposes.
        """
        keys = self.key_matrix(embeddings)
        queries = self.query_matrix(embeddings)
        values = self.value_matrix(embeddings)

        scores = queries @ keys.T
        weights = torch.softmax(scores / keys.shape[-1]**0.5, dim=-1)
        return weights @ values

