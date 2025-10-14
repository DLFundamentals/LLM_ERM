# collators.py
import torch
from typing import List, Dict

class PretrainedDataCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Set pad token if it's not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[tuple]) -> Dict[str, torch.Tensor]:
        # batch is a list of tuples, where each tuple is (sequence_tensor, label_tensor)
        
        sequences, labels = zip(*batch)
        
        # 1. Convert integer tensors to space-separated strings
        # We also strip the BOS token (index 0) which was added by your dataset
        text_sequences = [" ".join(map(str, seq[1:].tolist())) for seq in sequences]
        
        # 2. Tokenize the batch of strings
        tokenized_inputs = self.tokenizer(
            text_sequences,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 3. Add labels to the output dictionary
        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float)
        
        return tokenized_inputs