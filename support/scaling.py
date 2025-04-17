"""
Contains functions for scaling data.
"""
from typing import Callable
import pandas as pd
from sklearn.preprocessing import StandardScaler


to_scale = ['startYear', 'runtimeMinutes',
    'totalCredits', 'numRegions', 'ratingCount',
    'castNumber', 'companiesNumber', 'writerCredits',
    'directorsCredits', 'quotesTotal', 'totalMedia',
    'totalNominations', 'regions_freq_enc', 'regions_EU',
    'regions_NA', 'regions_AS', 'regions_AF', 'regions_OC', 'regions_SA',
    'regions_UNK', 'countryOfOrigin_freq_enc', 'countryOfOrigin_NA',
    'countryOfOrigin_AF', 'countryOfOrigin_AS', 'countryOfOrigin_EU',
    'countryOfOrigin_OC', 'countryOfOrigin_SA', 'countryOfOrigin_UNK',
    'reviewsTotal',
    'genre_Action', 'genre_Adult', 'genre_Adventure', 'genre_Animation',
    'genre_Biography', 'genre_Comedy', 'genre_Crime', 'genre_Documentary',
    'genre_Drama', 'genre_Family', 'genre_Fantasy', 'genre_Film-Noir',
    'genre_Game-Show', 'genre_History', 'genre_Horror', 'genre_Music',
    'genre_Musical', 'genre_Mystery', 'genre_News', 'genre_Reality-TV',
    'genre_Romance', 'genre_Sci-Fi', 'genre_Short', 'genre_Sport',
    'genre_Talk-Show', 'genre_Thriller', 'genre_War', 'genre_Western',
    'titleType_movie', 'titleType_short', 'titleType_tvEpisode',
    'titleType_tvMiniSeries', 'titleType_tvMovie', 'titleType_tvSeries',
    'titleType_tvShort', 'titleType_tvSpecial', 'titleType_video',
    'titleType_videoGame'
]

def scale_data(train: pd.DataFrame, test: pd.DataFrame | None=None, feats: list[str]=to_scale) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Scale the training and testing datasets using StandardScaler.

    Parameters:
        train (pd.DataFrame): The training DataFrame to scale.
        test (pd.DataFrame | None): The testing DataFrame to scale. If None, only the training DataFrame is processed.

        feats (list[str]): The features to scale. Defaults to a predefined list.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None]: The scaled training DataFrame and the scaled testing DataFrame (if provided).
    """
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform it
    train_scaled = train.copy()
    train_scaled[feats] = scaler.fit_transform(train[feats])

    # If a testing dataset is provided, transform it using the same scaler
    test_scaled = test.copy() if test is not None else None

    if test is not None:
        test_scaled[feats] = scaler.transform(test[feats])

    return train_scaled, test_scaled