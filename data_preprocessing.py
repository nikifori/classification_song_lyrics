"""
@File    :   data_preprocessing.py
@Time    :   12/2024
@Author  :   nikifori
@Version :   -
"""

import pandas as pd
from pathlib import Path
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split


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
    sampled_data = data.groupby('tag').apply(lambda x: x.sample(n=50000, random_state=random_state)).reset_index(drop=True)
    # sampled_data = data.groupby('tag').apply(lambda x: x.sample(n=1000, random_state=random_state)).reset_index(drop=True)
    train, temp = train_test_split(sampled_data, test_size=0.3, stratify=sampled_data['tag'], random_state=random_state)
    validation, test = train_test_split(temp, test_size=0.5, stratify=temp['tag'], random_state=random_state)
    return train, validation, test

def main():
    random_seed = 42
    set_random_state(random_seed)

    cleaned_data_path = Path(
        "/home/nikifori/Desktop/Master/NLP/final_project/classification_song_lyrics/data/song_lyrics_filtered.csv"
    )
    cleaned_data = pd.read_csv(cleaned_data_path)

    train, validation, test = split_data(
        cleaned_data,
        random_seed
    )

    print(1)


if __name__ == "__main__":
    main()
