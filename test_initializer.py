import torch 
from initializer import Initializer
from embedding import Embedding

def test_token_embeddings():
    """
    Here we load some data and then produce the embedding vectors from it.
    """
    ini = Initializer(max_length=4)
    dl = ini.input_dataloader_iter(batch_size=2, shuffle=False)
    inputs, targets = next(dl)
    print("Token IDs:\n", inputs)
    print("Size:\n", inputs.shape)
   
    emb = ini.embeddings(inputs)
    print("Input Embeddings:\n", emb)
    print("Input Embeddings size:\n", emb.shape)

    print("Full Positional embeddings:\n", ini.full_embeddings.pos)

