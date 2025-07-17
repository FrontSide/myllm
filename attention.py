import torch

class Attention():

    def __init__(self):
        pass

    def score_for_token_at_pos(self, embeddings, tok_idx):
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

