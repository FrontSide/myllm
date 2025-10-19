import torch
from attention_multi import MultiHeadAttentionWrapper 
import testutils

_TEST_EMBEDDINGS = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.1],
        [0.05, 0.8, 0.55]
    ])


def test_multi():
    """
    see page 84/85
    """
    torch.manual_seed(123)
    a = MultiHeadAttentionWrapper(token_vector_dim=3, weight_vector_dim=2, context_length=6, dropout=0.0, num_heads=2)
    input_batches = torch.stack((_TEST_EMBEDDINGS, _TEST_EMBEDDINGS), dim=0)
    context_matrix = a(input_batches) 
    assert context_matrix.shape == torch.Size([2, 6, 4]) # 2 batches, six words, 2 combined context vectors of dimesnion 2 each
    assert testutils.rounded(context_matrix)[1][2][:3] == [-0.63, -0.06, 0.62]
