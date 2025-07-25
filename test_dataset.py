
import torch
from initializer import Initializer

def test_dataloader():
    i = Initializer(max_length=4)
    dl = i.input_dataloader_iter(batch_size=1, shuffle=False)
    first_input, first_target = next(dl)
    assert first_input.tolist() == [[  40,  367, 2885, 1464]]
    assert first_target.tolist() == [[ 367, 2885, 1464, 1807]]
