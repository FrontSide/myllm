import torch
import pytest
from embedding import Embedding


def test_position_embedding():

    e = Embedding(vocab_size=5, output_dim=3, input_max_length=4)

    # The size of the position embedding must be imput_max_length * output_dim
    assert e.pos.shape == torch.Size([4, 3])

    # We should always get the exact same position vectors
    # (in this implementation, since we use a manual seed)  
    assert e.pos.tolist()[0][0] == 0.9728211164474487

def test_token_embedding():
    e = Embedding(vocab_size=5, output_dim=3, input_max_length=4)

    # The size of the position embedding must be vocab_size * output_dim
    assert e.weights().shape == torch.Size([5, 3])

    assert e._tok([2, 4]).tolist() == [[-0.9723550081253052, -0.755045473575592, 0.32390275597572327], [0.23497341573238373, 0.6652604341506958, 0.3528207540512085]]

    tks = e._tok([1, 4, 3, 1])
    # The tokens at index 0 and index 3 are both 1, therefore they must have the exact same token embeddings 
    assert tks.tolist()[0] == tks.tolist()[3]

def test_get_embedding():

    e = Embedding(vocab_size=5, output_dim=3, input_max_length=4)
   
    # make sure we enforce the correct length of the tokens array passed
    with pytest.raises(ValueError):
        e.get(tokens=[3, 1, 2]).tolist()

    tks = e.get(tokens=[1, 4, 3, 1])
    # the returned embeddings should be of shape len(tokens) * output_dim
    assert tks.shape == torch.Size([4, 3])

    # the tokens at index 0 and index 3 are both 1, however, because of their different position
    # they will not share the same embeddings
    assert tks.tolist()[0][0] != tks.tolist()[3][0]
