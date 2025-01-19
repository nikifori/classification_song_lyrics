'''
@File    :   logistic_regression.py
@Time    :   01/2025
@Author  :   nikifori
@Version :   -
'''
from pathlib import Path
import pandas as pd
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions
import nltk
import re

# Ensure NLTK resources are downloaded
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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
    lyrics = re.sub(r'\[.*?\]', '', lyrics)

    # Remove newline characters
    lyrics = re.sub(r'\n+', '. ', lyrics)

    # Remove special characters except alphanumeric and basic punctuation
    lyrics = re.sub(r'[^a-zA-Z0-9.,!?\'" ]', '', lyrics)

    # Replace multiple spaces with a single space
    lyrics = re.sub(r'\s+', ' ', lyrics)

    # Remove ". " at the beginning of the text if present
    lyrics = re.sub(r'^\.\s*', '', lyrics)

    # Strip leading and trailing spaces
    lyrics = lyrics.strip()

    # Replace multiple periods with a single period
    lyrics = re.sub(r'\.\.+', '.', lyrics)

    return lyrics

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

def apply_preprocessing(df):
    df["lyrics"] = df["lyrics"].apply(preprocess_text)
    return df

def create_tfidf_features(train, validation, test):
    tfidf = TfidfVectorizer(max_features=10000)

    X_train = tfidf.fit_transform(train["lyrics"])
    X_validation = tfidf.transform(validation["lyrics"])
    X_test = tfidf.transform(test["lyrics"])

    y_train = train["tag"]
    y_validation = validation["tag"]
    y_test = test["tag"]

    return X_train, y_train, X_validation, y_validation, X_test, y_test, tfidf

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


def main():
    csv_path = "/home/nikifori/Desktop/Master/NLP/final_project/classification_song_lyrics/data/song_lyrics_filtered.csv"
    # csv_path = "/home/nikifori/Desktop/Master/NLP/final_project/classification_song_lyrics/data/song_lyrics_filtered_debugging.csv"
    random_seed = 42

    exp_folder = Path("/home/nikifori/Desktop/Master/NLP/final_project/classification_song_lyrics/exps/logistic_regression")
    exp_folder.mkdir(parents=True, exist_ok=True)

    csv_path = Path(csv_path)
    assert csv_path.is_file(), "csv_path must be a valid file"

    data = pd.read_csv(csv_path)
    n_classes = len(data["tag"].unique())

    set_random_state(random_seed)
    train, validation, test = split_data(data, random_seed)

    train = apply_preprocessing(train)
    validation = apply_preprocessing(validation)
    test = apply_preprocessing(test)

    # tfidf = TfidfVectorizer(max_features=10000)
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(train["lyrics"])
    X_validation = tfidf.transform(validation["lyrics"])
    X_test = tfidf.transform(test["lyrics"])

    y_train = train["tag"]
    y_validation = validation["tag"]
    y_test = test["tag"]

    model = LogisticRegression(max_iter=10000, random_state=random_seed, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    labels = sorted(data["tag"].unique())

    report = classification_report(y_test, y_pred, target_names=labels)
    confusion = confusion_matrix(y_test, y_pred)

    # Save classification report
    exp_folder.mkdir(parents=True, exist_ok=True)
    report_path = exp_folder / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Save confusion matrix
    confusion_path = exp_folder / "confusion_matrix.txt"
    with open(confusion_path, "w") as f:
        f.write(str(confusion))

    # Plot and save confusion matrix using sklearn
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    figure_path = exp_folder / "confusion_matrix.png"
    plt.savefig(figure_path)
    plt.close()



if __name__ == '__main__':
    main()