from initializer import Initializer
import torch

def test_initializer():
    ini = Initializer(max_length=4)
    dl = ini.input_dataloader_iter(batch_size=2, shuffle=False)
    inputs, targets = next(dl)

    assert inputs[0].tolist() == [40, 367, 2885, 1464] 
    assert targets[0].tolist() == [367, 2885, 1464, 1807]

    assert inputs.shape == torch.Size([2, 4])
    assert targets.shape == torch.Size([2, 4])
   
    emb = ini.embeddings(inputs)
    print("Input Embeddings:\n", emb)
    print("Input Embeddings size:\n", emb.shape)

    print("Full Positional embeddings:\n", ini.full_embeddings.pos)

    assert emb.shape == torch.Size([2, 4, 256])


def test_known_sequence():

    ini = Initializer(max_length=6, embedding_dim=3)

    tks = ini.tokenizer.encode("Your Journey Starts With One Step")

    assert tks == [7120, 15120, 50181, 2080, 1881, 5012] 

    #emb = ini.embeddings(tks)

