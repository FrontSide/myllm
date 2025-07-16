
import torch
from initializer import Initializer

def test_dataloader():
    i = Initializer(max_length=4)
    dl = i.input_dataloader_iter(batch_size=1, shuffle=False)
    first_batch = next(dl)
    assert first_batch == [torch.tensor([[  40,  367, 2885, 1464]]), torch.tensor([[ 367, 2885, 1464, 1807]])]
