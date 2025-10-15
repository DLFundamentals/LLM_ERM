# collators.py

import torch
from typing import List, Dict
from transformers.tokenization_utils_base import BatchEncoding

class PretrainedDataCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Set pad token if it's not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[tuple]) -> Dict[str, torch.Tensor]:
        # batch is a list of tuples, where each tuple is (sequence_tensor, label_tensor)
        
        sequences = []
        labels = []
        for item in batch:
            data, label = item
            
            seq = data[0] if isinstance(data, tuple) else data
            sequences.append(seq.detach().cpu())
            
            labels.append(label)

        # 1. Convert integer tensors to space-separated strings
        text_sequences = [" ".join(map(str, seq[1:].tolist())) for seq in sequences]
        
        # 2. Tokenize, but don't return tensors immediately to handle version differences
        tokenized_output = self.tokenizer(
            text_sequences,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors=None  # Get lists of IDs first
        )
        
        # 3. Manually construct the batch dictionary to ensure a consistent output type
        # This is robust against older transformers versions that might return a list.
        if isinstance(tokenized_output, (dict, BatchEncoding)):
            # Modern transformers versions return a dict-like BatchEncoding
            input_ids = torch.tensor(tokenized_output['input_ids'])
            attention_mask = torch.tensor(tokenized_output['attention_mask'])
        else: 
            # Older versions might return a list of Encoding objects.
            # We manually extract the data and convert it to tensors.
            input_ids_list = [enc.ids for enc in tokenized_output]
            attention_mask_list = [enc.attention_mask for enc in tokenized_output]
            input_ids = torch.tensor(input_ids_list)
            attention_mask = torch.tensor(attention_mask_list)

        # 4. Assemble the final, guaranteed-to-be-a-dict batch
        final_batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.stack([torch.as_tensor(l) for l in labels]).float()
        }
        
        return final_batch