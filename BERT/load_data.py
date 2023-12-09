import pandas as pd
import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader


class LoadData:
    """This class loads data to a specific format from a working folder."""
    def __init__(self, file):
        self.data = file

    def load(self, batch_size=32, shuffle=True):
        """This function loads data."""

        # Load data file into a pandas df
        df = pd.read_csv(self.data, sep='\t')

        # Drop NA rows
        df = df.dropna()

        # Remove all rows containing '-'
        df = df[~df['sentence1'].str.contains('-')]
        df = df[~df['sentence2'].str.contains('-')]
        df = df[~df['gold_label'].str.contains('-')]

        # Specify labels
        labels = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

        # Initialize a tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        token_ids = []
        mask_ids = []
        segments_ids = []
        y = []

        # Convert required columns from df to lists
        text_list = df['sentence1'].to_list()
        hypothesis_list = df['sentence2'].to_list()
        label_list = df['gold_label'].to_list()

        # Iterate through above created lists in order to create token ids, segment ids and mask ids
        for (text, hypothesis, label) in zip(text_list, hypothesis_list, label_list):
            # Tokenize text sentence
            text_id = tokenizer.encode(text, add_special_tokens=False)
            # Tokenize hypothesis sentence
            hypothesis_id = tokenizer.encode(hypothesis, add_special_tokens=False)
            # CLS + text + SEP + hypothesis + SEP
            pair_token_ids = torch.tensor(
                [tokenizer.cls_token_id] + text_id + [tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id]
            )
            # Create segment ids (each sentence is given its id and their ids tokens are changed by ids)
            segment_ids = torch.tensor([0] * (len(text_id) + 2) + [1] * (len(hypothesis_id) + 1))
            # Create attention mask ids (change all sentence tokens to 1's, else to 0's)
            attention_mask_ids = torch.tensor([1] * (len(text_id) + len(hypothesis_id) + 3))
            # Append results to above created lists
            token_ids.append(pair_token_ids)
            segments_ids.append(segment_ids)
            mask_ids.append(attention_mask_ids)
            y.append(labels[label])

        # Pad sequences (make sure that all sequences are of the same length by means of adding 0's to shorter ones)
        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        segments_ids = pad_sequence(segments_ids, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, segments_ids, y)

        loader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size
        )

        return loader
