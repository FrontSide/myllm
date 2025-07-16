import tiktoken

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
        return self.tokenizer.decode(tokens)
