
from torch.utils.data import DataLoader
from embedding import Embedding
from tokenizer import Tokenizer
from dataset import GPTDataset

class Initializer():

    """
    Initializes the Input data. This includes:
    - Reading in the raw input data from a text file
    - Converting the data into tokens
    - Loading the data into a torch Dataloader (to create token tensors and for easier data handling)
    - Creating embedding vectors

    max_length is the maximum length/number of tokens in an input set (and target set, respectively) used for training.
    embeding_dim is the number of dimensions for an embedding vector, i.e. each token will receive a vector of this size
    """

    DEFAULT_TEXT = "./data/the-verdict.txt"

    def __init__(self, max_length=4, embedding_dim=256):
        self.max_length = max_length
        self.txt = Initializer.get_raw_input_text()
        self.tokenizer = Tokenizer()
        self.full_embeddings = Embedding(self.tokenizer.vocab_size, output_dim=embedding_dim, input_max_length=max_length)
        self.dataset = GPTDataset(self.txt, self.tokenizer, self.max_length, stride=self.max_length)

    @staticmethod
    def get_raw_input_text():
        with open(Initializer.DEFAULT_TEXT, "r", encoding="utf-8") as f:
            return f.read()

    def input_dataloader_iter(self, batch_size=4, shuffle=True, drop_last=True, num_workers=0):
        return self.dataset.dataloader_iter(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
       
    def embeddings(self, tokens):
        """
        Returns the embedinggs tensor for the given tokens.
        See Embeddings.embeddings() for more info
        """
        return self.full_embeddings.get(tokens)

