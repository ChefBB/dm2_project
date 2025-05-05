import pandas as pd
from collections import Counter

def process_genre_features(df):
    # Step 1: Convert genres to list
    df['genres_list'] = df['genres'].fillna('').apply(lambda x: x.split(',')[:3])

    # Step 2: Count frequencies
    all_genres = df['genres_list'].sum()
    genre_freq = Counter(all_genres)

    #setting the frequency of '\\N' to 0
    genre_freq['\\N'] = 0

    # Step 3: Sort each row’s genres by overall frequency
    def sort_genres_by_frequency(genres):
        return sorted(genres, key=lambda g: -genre_freq.get(g, 0))

    df['sorted_genres'] = df['genres_list'].apply(sort_genres_by_frequency)

    # Step 4: Assign frequencies
    df['genre1'] = df['sorted_genres'].apply(lambda x: genre_freq.get(x[0], 0) if len(x) > 0 else 0)
    df['genre2'] = df['sorted_genres'].apply(lambda x: genre_freq.get(x[1], 0) if len(x) > 1 else 0)
    df['genre3'] = df['sorted_genres'].apply(lambda x: genre_freq.get(x[2], 0) if len(x) > 2 else 0)

    # Step 5: Drop intermediate columns
    df.drop(columns=['genres', 'genres_list', 'sorted_genres'], inplace=True)

    return df
