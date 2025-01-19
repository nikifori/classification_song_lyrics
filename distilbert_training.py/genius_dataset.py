"""
@File    :   genius_dataset.py
@Time    :   12/2024
@Author  :   nikifori
@Version :   -
"""

import torch
from pathlib import Path
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from transformers import AutoTokenizer
import re


def set_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_lyrics(lyrics):
    # Remove content in brackets (e.g., [Chorus], [Verse 1])
    lyrics = re.sub(r"\[.*?\]", "", lyrics)

    # Remove newline characters
    lyrics = re.sub(r"\n+", ". ", lyrics)

    # Remove special characters except alphanumeric and basic punctuation
    lyrics = re.sub(r'[^a-zA-Z0-9.,!?\'" ]', "", lyrics)

    # Replace multiple spaces with a single space
    lyrics = re.sub(r"\s+", " ", lyrics)

    # Remove ". " at the beginning of the text if present
    lyrics = re.sub(r"^\.\s*", "", lyrics)

    # Strip leading and trailing spaces
    lyrics = lyrics.strip()

    return lyrics


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
    sampled_data = data.groupby('tag').apply(lambda x: x.sample(
        n=50000, random_state=random_state)).reset_index(drop=True)
    # sampled_data = (
    #     data.groupby("tag")
    #     .apply(lambda x: x.sample(n=1000, random_state=random_state))
    #     .reset_index(drop=True)
    # )
    sampled_data["lyrics"] = sampled_data["lyrics"].apply(preprocess_lyrics)
    train, temp = train_test_split(
        sampled_data,
        test_size=0.3,
        stratify=sampled_data["tag"],
        random_state=random_state,
    )
    validation, test = train_test_split(
        temp, test_size=0.5, stratify=temp["tag"], random_state=random_state
    )
    return train, validation, test


class Genius_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str = None,
        random_seed: int = 42,
        transformer_model: str = "distilbert-base-uncased",
        split_name: str = "train",
    ):
        # If seed is the same along all initializations of Genius_dataset
        # self.train, self.validation, self.test are certain not to contain
        # any mutual data.
        valid_splits = ["train", "validation", "test"]
        if split_name not in valid_splits:
            raise ValueError(f"Invalid split value. Must be one of {valid_splits}.")
        self.split_name = split_name

        assert csv_path is not None, "csv_path must be specified"

        csv_path = Path(csv_path)
        assert csv_path.is_file(), "csv_path must be a valid file"

        self.data = pd.read_csv(csv_path)
        self.n_classes = len(self.data["tag"].unique())

        set_random_state(random_seed)
        self.train, self.validation, self.test = split_data(self.data, random_seed)

        if self.split_name == "train":
            self.dataset = self.train
        elif self.split_name == "validation":
            self.dataset = self.validation
        else:
            self.dataset = self.test

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset.iloc[idx]["lyrics"]
        y = self.dataset.iloc[idx]["tag"]
        y = ["rap", "pop", "rock", "country", "rb"].index(y)  # Convert tag to index
        encoded_input = self.tokenizer(
            x,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "label": torch.tensor(y),
        }


def main():
    pass


if __name__ == "__main__":
    main()
