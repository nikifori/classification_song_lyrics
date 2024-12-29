'''
@File    :   genius_dataset.py
@Time    :   12/2024
@Author  :   nikifori
@Version :   -
'''
import torch
from pathlib import Path
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer


def set_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_data(data, random_state):
    """_summary_
    Splits the data into 70% training, 15% evaluation and 15% testing.
    Stratification == True

    Args:
        data (_type_): _description_
        random_state (_type_): _description_

    Returns:
        _type_: _description_
    """
    sampled_data = data.groupby('tag').apply(lambda x: x.sample(n=50000, random_state=random_state)).reset_index(drop=True)
    train, temp = train_test_split(sampled_data, test_size=0.3, stratify=sampled_data['tag'], random_state=random_state)
    validation, test = train_test_split(temp, test_size=0.5, stratify=temp['tag'], random_state=random_state)
    return train, validation, test

class Genius_trainig_dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            csv_path: str = None,
            random_seed: int = 42,
            transformer_model: str = "distilbert-base-uncased"
    ):
        assert csv_path is not None, "csv_path must be specified"

        csv_path = Path(csv_path)
        assert csv_path.is_file(), "csv_path must be a valid file"

        self.data = pd.read_csv(csv_path)
        self._split = "train" # ["train", "validation", "test"]
        self.n_classes = len(self.data["tag"].unique())

        set_random_state(random_seed)
        self.train, self.validation, self.test = split_data(
            self.data,
            random_seed
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained(transformer_model)
    
    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, value):
        valid_splits = ["train", "validation", "test"]
        if value not in valid_splits:
            raise ValueError(f"Invalid split value. Must be one of {valid_splits}.")
        self._split = value
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self._split == "train":
            ...
        elif self._split == "validation":
            ...
        else:
            ...


def main():
    pass


if __name__ == '__main__':
    main()