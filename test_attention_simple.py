import torch
from attention_simple import AttentionSimple

_TEST_EMBEDDINGS = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.1],
        [0.05, 0.8, 0.55]
    ])



def test_score_for_token():

    """
    see page57/58 in book
    """

    a = AttentionSimple()
    assert [round(x, 2) for x in a.scores_for_token(_TEST_EMBEDDINGS, 1).tolist()] == [0.95, 1.50, 1.48, 0.84, 0.71, 1.09]

def test_weights_for_token():

    """
    see page 60
    """
    a = AttentionSimple()
    assert [round(x, 2) for x in a.weights_for_token(_TEST_EMBEDDINGS, 1).tolist()] == [0.14, 0.24, 0.23, 0.12, 0.11, 0.16]


def test_context_vector_for_token():

    """
    see page 60
    """

    a = AttentionSimple()
    assert [round(x, 2) for x in a.context_vector_for_token(_TEST_EMBEDDINGS, 1).tolist()] == [0.44, 0.65, 0.57]

def test_scores():
    """
    see page 62
    """

    a = AttentionSimple()
    scores = a.scores(_TEST_EMBEDDINGS)
    assert scores.shape == torch.Size([6, 6])
    assert round(scores[1][1].item(), 4) == 1.4950

def test_weights():
    """
    see page 63
    """

    a = AttentionSimple()
    weights = a.weights(_TEST_EMBEDDINGS)
    assert weights.shape == torch.Size([6, 6])
    assert round(weights[1][1].item(), 4) == 0.2379

def test_context_vectors():
    """
    see page 63
    """

    a = AttentionSimple()
    context_vectors = a.context_vectors(_TEST_EMBEDDINGS)
    assert context_vectors.shape == torch.Size([6, 3])
    assert round(context_vectors[1][1].item(), 4) == 0.6515
    assert context_vectors[2].tolist() == a.context_vector_for_token(_TEST_EMBEDDINGS, 2).tolist()
