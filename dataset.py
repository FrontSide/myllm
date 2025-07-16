from torch.utils.data import Dataset, DataLoader
import torch

class GPTDataset(Dataset):

    """
    The GPTDataset builds a training dataset out of the input text (txt).
    One training pair, looks like this (but tokenized):
        input_chunk  = ["This", "is", "an", "example"]
        target_chunk = ["is", "an", "example", "sentence"]
    
    max_length is effectively the number of tokens in the input chunk,
      the output chunk will be the same length with an offset of one token,
      therefore including the one token following the input sequence. 

    stride defines how far we skip ahead (from the first word of the previous input_cunk) to start the next input_chunk.
      if stride is equal to max_length, we will have all text encompassed in the training data without any overlap.
    """

    def __init__(self, txt, tokenizer, max_length, stride):

        self.input_ids = []
        self.target_ids = []

        input_text_tokens = tokenizer.encode(txt)

        for i in range(0, len(input_text_tokens) - max_length, stride):
            input_chunk = input_text_tokens[i:i + max_length]
            target_chunk = input_text_tokens[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


    def dataloader(self, batch_size=4, shuffle=True, drop_last=True, num_workers=0):
        """
        Will create a dataloader that created training data in batches from the given raw text (self.txt).
        See GPTDataset for more information.

        batch_size is the number of input-target pairs the dataloader will store per "batch".
         So e.g. if batch_size=2 and you access the first batch e.g. with next(iter(dataloader)) you will get a tensor 
         of size 2 x max_length.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    def dataloader_iter(self, batch_size=4, shuffle=True, drop_last=True, num_workers=0):
        return iter(self.dataloader(batch_size, shuffle, drop_last, num_workers))


