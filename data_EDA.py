'''
@File    :   data_EDA.py
@Time    :   01/2025
@Author  :   nikifori
@Version :   -
'''
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import random
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Ensure NLTK resources are downloaded
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Expand contractions
    text = contractions.fix(text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    text = " ".join(tokens)
    return text

# def apply_preprocessing(df):
#     df["lyrics"] = df["lyrics"].apply(preprocess_text)
#     return df

# Helper function to apply preprocessing in parallel
def preprocess_parallel(row):
    return preprocess_text(row)

def apply_preprocessing(df, num_cores=1):
    # Initialize tqdm for pandas apply
    tqdm.pandas()

    # Use ProcessPoolExecutor to parallelize the processing across multiple cores
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Map the function to the rows of the "lyrics" column in parallel
        df["lyrics"] = list(tqdm(executor.map(preprocess_parallel, df["lyrics"]), total=len(df)))
    
    return df

def preprocess_lyrics(lyrics):
    # Remove content in brackets (e.g., [Chorus], [Verse 1])
    lyrics = re.sub(r"\[.*?\]", "", lyrics)

    # Remove newline characters
    lyrics = re.sub(r"\n+", ". ", lyrics)

    # Remove special characters except spaces and alphanumeric characters
    lyrics = re.sub(r'[^a-zA-Z0-9 ]', "", lyrics)

    # Replace multiple spaces with a single space
    lyrics = re.sub(r"\s+", " ", lyrics)

    # Remove ". " at the beginning of the text if present
    lyrics = re.sub(r"^\.\s*", "", lyrics)

    # Strip leading and trailing spaces
    lyrics = lyrics.strip()

    return lyrics

def sample_data(data, random_state):
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
    sampled_data["lyrics"] = sampled_data["lyrics"].apply(preprocess_lyrics)

    return sampled_data

def set_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_top_words_per_class(df, column, tag_column, top_n=10):
    top_words = {}
    for tag in df[tag_column].unique():
        words = df[df[tag_column] == tag][column].str.cat(sep=" ").split()
        word_freq = pd.Series(words).value_counts()
        top_words[tag] = word_freq.head(top_n).to_dict()
    return top_words

def get_unique_word_counts(df, column, tag_column):
    unique_word_counts = {}
    for tag in df[tag_column].unique():
        words = set(df[df[tag_column] == tag][column].str.cat(sep=" ").split())
        unique_word_counts[tag] = len(words)
    return unique_word_counts

def main():
    csv_path = "/home/nikifori/Desktop/Master/NLP/final_project/classification_song_lyrics/data/song_lyrics_filtered.csv"
    csv_path = Path(csv_path)
    random_seed = 42
    num_cores = 16
    output_dir = Path("/home/nikifori/Desktop/Master/NLP/final_project/data_insight")
    output_dir.mkdir(exist_ok=True)

    data = pd.read_csv(csv_path)
    _classes = len(data["tag"].unique())
    set_random_state(random_seed)

    data = sample_data(data, random_seed)
    data = apply_preprocessing(data, num_cores=num_cores)

    # Top words per class
    top_words = get_top_words_per_class(data, "lyrics", "tag", top_n=50)
    with open(output_dir / "top_words_per_class.json", "w") as f:
        import json
        json.dump(top_words, f, indent=4)

    # Unique word counts per class
    unique_word_counts = get_unique_word_counts(data, "lyrics", "tag")
    with open(output_dir / "unique_word_counts_per_class.json", "w") as f:
        json.dump(unique_word_counts, f, indent=4)

    print("Analysis completed. Results saved to the 'output' directory.")

    print(1)


if __name__ == '__main__':
    main()