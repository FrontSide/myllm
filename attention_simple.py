import torch

class AttentionSimple():

    def __init__(self):
        pass

    def scores_for_token(self, embeddings, tok_idx):
        """
        Calculate the attention scores of all tokens in embeddings
        relative to the "attended" token at index tok_idx
      
        The attention values are the dot products of the attended token and each other token.

        So for example if the input embeddings are these 3 tokens (with 3-dim embedding vectors):
            embeddings = [
                [0.1, 0.2, 0.4],
                [-0.2, 0.3, 0.2],
                [-0.1, -0.1, 0.3]
            ]
        and we want to attend to token at index 1 we get a result of 
            [
                [0.1, 0.2, 0.4] dot [-0.2, 0.3, 0.2], 
                [-0.2, 0.3, 0.2] dot [-0.2, 0.3, 0.2], 
                [-0.1, -0.1, 0.3] dot [-0.2, 0.3, 0.2], 

        params:
            embeddings must be a tensor of order 2,
                with each row being a vector of embeddings represneting one token.
        """

        query = embeddings[tok_idx]
        attn_scores = torch.empty(embeddings.shape[0])
        for i, x_i in enumerate(embeddings):
            attn_scores[i] = torch.dot(x_i, query)
        return attn_scores

    def weights_for_token(self, embeddings, tok_idx):
        """
        Calculates the attention weights of all tokens in embeddints 
        relative to the "attended" token at tok_idx

        Attention weights are normalized attention scores
        We use torch.softmax for normalization
        """
        attention_scores = self.scores_for_token(embeddings, tok_idx)
        return torch.softmax(attention_scores, dim=0)

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
