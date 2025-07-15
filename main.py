
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


sampletext = "This is atrsg simple sample text."

class Tokenizer:

    """
    We use the tiktoken Byte-Pair Encoding tokenizer
    """

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = 50257 # The vocabulary size of the gpt2 encoding

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        self.tokenizer.decode(tokens)

class GPTDataset(Dataset):

    """
    The GPTDataset builds a training dataset out of the input text (txt).
    One training pair, looks like this (but tokenized):
        input_chunk  = ["This", "is", "an", "example"]
        target_chunk = ["is", "an", "example", "sentence"]
    
    max_length is effectively the number of tokens in the input chunk,
      the output chunk will be the same length with an offset of one token,
      therefore including the one token following the input sequence. 

    stride defines how far we skip ahead (from the first word of the previous input_cunk) to start the next input_chunk.
      if stride is equal to max_length, we will have all text encompassed in the training data without any overlap.
    """

    def __init__(self, txt, tokenizer, max_length, stride):

        self.input_ids = []
        self.target_ids = []

        input_text_tokens = tokenizer.encode(txt)

        for i in range(0, len(input_text_tokens) - max_length, stride):
            input_chunk = input_text_tokens[i:i + max_length]
            target_chunk = input_text_tokens[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


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
        self.token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        self.pos_embedding_layer = torch.nn.Embedding(input_max_length, output_dim)
        self.pos = self.pos_embedding_layer(torch.arange(input_max_length))

    def weights(self):
        return self.token_embedding_layer.weight

    def get(self, tokens):
        """
        Returns the embedding tensor including the positional embeddings for the given tokens.

        The position tensor is a fixed tensor created in the constructor above.
        It is a 2-dimensional tensor of size input_max_length * output_dim, meaning that there will be 
        one vector (of output_dim dimensions) for every token position, whereas there are at most input_max_length tokens in the input.
        The position tensor will then be added to each set of tokens given here.

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
    

class Initializer():

    """
    Initializes the Input data. This includes:
    - Reading in the raw input data from a text file
    - Converting the data into tokens
    - Loading the data into a torch Dataloader (to create token tensors and for easier data handling)
    - Creating embedding vectors

    max_length is the maximum length/number of tokens in an input set (and target set, respectively) used for training.
    """

    def __init__(self, max_length):
        self.max_length = max_length
        self.txt = Initializer.get_raw_input_text()
        self.tokenizer = Tokenizer()
        self.full_embeddings = Embedding(self.tokenizer.vocab_size, output_dim=256, input_max_length=max_length)

    @staticmethod
    def get_raw_input_text():
        with open("./data/the-verdict.txt", "r", encoding="utf-8") as f:
            return f.read()

    def input_dataloader(self, batch_size=4, stride=128, shuffle=True, drop_last=True, num_workers=0):
        """
        Will create a dataloader that created training data in batches from the given raw text (self.txt).
        See GPTDataset for more information.

        batch_size is the number of input-target pairs the dataloader will store per "batch".
         So e.g. if batch_size=2 and you access the first batch e.g. with next(iter(dataloader)) you will get a tensor 
         of size 2 x max_length.
        """
        dataset = GPTDataset(self.txt, self.tokenizer, self.max_length, stride)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    def input_dataloader_iter(self, batch_size=4, stride=128, shuffle=True, drop_last=True, num_workers=0):
        return iter(self.input_dataloader(batch_size, stride, shuffle, drop_last, num_workers))
       
    def embeddings(self, tokens):
        """
        Returns the embedinggs tensor for the given tokens.
        See Embeddings.embeddings() for more info
        """
        return self.full_embeddings.get(tokens)

def test_dataloader():
    dl = Initializer(max_length=4).input_dataloader_iter(batch_size=1, stride=1, shuffle=False)
    first_batch = next(dl)
    print(first_batch)

def test_embedding_weights():
    e = Embedding(5, 3, input_max_length=4)
    print(e.weights())
    print(e.get([2, 5]))

def test_token_embeddings():
    """
    Here we load some data and then produce the embedding vectors from it.
    """
    ini = Initializer(max_length=4)
    dl = ini.input_dataloader_iter(batch_size=2, stride=1, shuffle=False)
    inputs, targets = next(dl)
    print("Token IDs:\n", inputs)
    print("Size:\n", inputs.shape)
   
    emb = ini.embeddings(inputs)
    print("Input Embeddings:\n", emb)
    print("Input Embeddings size:\n", emb.shape)

    print("Full Positional embeddings:\n", ini.full_embeddings.pos)

test_token_embeddings()
