import torch
from attention import Attention

def test_score_at_position():

    """
    see page57/58 in book
    """

    a = Attention()

    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.1],
        [0.05, 0.8, 0.55]
    ])

    assert [round(x, 2) for x in a.score_for_token_at_pos(inputs, 1).tolist()] == [0.95, 1.50, 1.48, 0.84, 0.71, 1.09]



