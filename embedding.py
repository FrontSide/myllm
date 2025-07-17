import torch

class Embedding():

    """
    Tokens need to be converted into vectors so that they can be fed into the neural network that is the LLM.
    In order to do this we start by creating a tensor or random weights. The weights will be a 2-dimensional matrix
    of size vocab_size x output_dim.

    In addition to these token embeddings we will also create position embeddings which are needed to make the LLM 
    aware of the position of a token within the input to improve the LLMs qualitative performance
    as the position of a word within a sentence is a crucial piece of information in determining what word should follow a sequence.

    vocab_size it the number of tokens in the vocabulary created by the tokenizer 
    output_dim is the number of dimensions of the embedding weights vector. Each token will have output_dim weights.

    input_max_length is the maximum amount of tokens in an input sequence which is needed for the positional embeddings 
      and may be set to the same value as the max_length when loading the data
    """

    def __init__(self, vocab_size, output_dim, input_max_length):
        torch.manual_seed(123)
        self.input_max_length = input_max_length
        self.token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        self.pos_embedding_layer = torch.nn.Embedding(self.input_max_length, output_dim)
        self.pos = self.pos_embedding_layer(torch.arange(self.input_max_length))

    def weights(self):
        return self.token_embedding_layer.weight

    def get(self, tokens):
        """
        Returns the embedding tensor including the positional embeddings for the given tokens.

        The position tensor is a fixed tensor created in the constructor above.
        It is a 2-dimensional tensor of size input_max_length * output_dim, meaning that there will be 
        one vector (of output_dim dimensions) for every token position, whereas there are at most input_max_length tokens in the input.
        The position tensor will then be added to each set of tokens given here.

        For this to work, the number of tokens per batch passed to this method must be of size input_max_length

        For example. Let's say we want to retrieve the embeddings for two sets of tokens, whereas the (input_)max_length is set to 3
            tokens = tensor([3, 45, 23], [76, 23, 12])
    
        Given output_dim of 4, we will get a 4-dimensional vector for each individual token taken from the "_tok" token embedding layer.
            _tok(tokens) = tensor(
                [
                    [1.2, -1.1, 0.7, 0.5] //embeddings for token 3
                    [1.4, 1.1, 0.2, -0.4] //embeddings for token 45
                    [...] //etc...
                ],
                [
                    [1.1, -1.1, 0.9, -0.8] //embeddings for token 76 in the second set
                    [][] //etc...
                ] 
            )

        Now we will also retrieve the positional embeddings, which may look like this:
            pos = tensor(
                [-0.2, 1.2, -1.2, 0.4] //embeddings for position 1
                [] //etc...
                []
            )

        The two tensors will now be added together so that each token vector will get added the respective position vector 

            _tok(tokend) + self.pos = tensor(
                [
                    [1.0, 0.1, -0.5, 0.9] //addition of embeddings for token 3 and embeddings for position 1 
                    [] //addition of embeddings for token 45 and embeddings for position 2
                    [] //etc
                ],
                [
                    [0.9, 0.1, -0.3, 1.3] //addition of embeddings for token 76 in second set and embeddings for position 1 again
                    []
                    []
                ]
            )
        """
    
        tokens_tensor = torch.tensor(tokens)
        if tokens_tensor.ndim == 2 and tokens_tensor.size()[1] != self.input_max_length:
            raise ValueError(f"number of tokens per batch must be {self.input_max_length}, was {tokens_tensor.size()[1]}")
        if tokens_tensor.ndim == 1 and len(tokens_tensor) != self.input_max_length:
            raise ValueError(f"number of tokens must be {self.input_max_length}, was {len(tokens_tensor)}")
        if tokens_tensor.ndim > 2:
            raise ValueError(f"invalid number of dimensiond for passed tokens. Must be 1 or 2, was {tokens_tensor.ndim}")

        return self._tok(tokens) + self.pos 

    def _tok(self, tokens):
        """
        Returns the embedding tensor WITHOUT the positional tensor for the given tokens 
        Note that this is simply an index lookup,
        so if tokens=[5, 7], we will simply return the vectors at index 5 and 7, respectively,
        from the weights matrix.

        Note that tokens can either be a simple array of tokens (i.e. a single vector) or a 2-dimensional torch tensor,
        where each row is a batch of tokens
        """
        return self.token_embedding_layer(torch.tensor(tokens)) 
    


