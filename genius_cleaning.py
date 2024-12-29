'''
@File    :   genius_cleaning.py
@Time    :   12/2024
@Author  :   nikifori
@Version :   -
'''
import pandas as pd


def main():
    df = pd.read_csv("/home/nikifori/Desktop/Master/NLP/final_project/data/song_lyrics.csv")
    df_filtered =df[df['language'].notna()]
    df_filtered = df_filtered[df_filtered['language'] == 'en'] # English songs only selection
    genre_counts = df_filtered['tag'].value_counts()

    df_filtered = df_filtered[df_filtered["year"] > 1960]
    selected_columns = ['title', 'tag', 'views', 'lyrics', 'id']
    df_filtered = df_filtered[selected_columns]
    df_filtered = df_filtered[df_filtered["tag"] != "misc"]
    df_filtered = df_filtered[df_filtered['lyrics'].apply(lambda x: len(x.split(" ")) > 150)]
    # df_filtered = df_filtered[df_filtered["views"] > 1000]
    print(1)


if __name__ == '__main__':
    main()