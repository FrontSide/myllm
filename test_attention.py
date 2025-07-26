import torch
from attention import Attention

_TEST_EMBEDDINGS = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.1],
        [0.05, 0.8, 0.55]
    ])



def test_qkv_matrices():

    """
    see page66 in book
    """

    a = Attention(token_vector_dim=3, weight_vector_dim=2)

    assert [round(x, 4) for x in a.query_matrix(torch.tensor(_TEST_EMBEDDINGS[1])).tolist()] == [0.4306, 1.4551]

    # Each matrix will have one row per input token, whereas each row is a vector of dimension weight_vector_dim
    assert a.query_matrix(_TEST_EMBEDDINGS).shape == torch.Size([6, 2])
    assert a.key_matrix(_TEST_EMBEDDINGS).shape == torch.Size([6, 2])
    assert a.value_matrix(_TEST_EMBEDDINGS).shape == torch.Size([6, 2])

def test_attention_scores_for_token():

    """
    see page67/68 in book
    """
    a = Attention(token_vector_dim=3, weight_vector_dim=2)

    # Calculate the attention scores of all tokens relative to the attended to token at idx 1
    scores = a.scores_for_token(_TEST_EMBEDDINGS, 1)
    assert scores.shape == torch.Size([6])
    assert [round(x, 4) for x in scores.tolist()] == [1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]

def test_attention_weights_for_token():
    """
    see page68
    """
    a = Attention(token_vector_dim=3, weight_vector_dim=2)

    # Calculate the attention weights of all tokens relative to the attended to token at idx 1
    weights = a.weights_for_token(_TEST_EMBEDDINGS, 1)
    assert weights.shape == torch.Size([6])
    assert sum(weights) == 1
    assert [round(x, 4) for x in weights.tolist()] == [0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]
