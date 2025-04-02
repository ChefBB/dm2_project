"""
This file contains various preprocessing transformations for
the dataset
"""
import pandas as pd
import numpy as np


to_log = [
    'criticReviewsTotal',
    'numRegions', 'userReviewsTotal', 'ratingCount',
    'castNumber', 'companiesNumber', 'externalLinks',
    'writerCredits', 'directorsCredits', 'totalMedia',
    'totalNominations',
    # 'regions_freq_enc', 'regions_EU', 'regions_NA', 'regions_AS',
    # 'regions_AF', 'regions_OC', 'regions_SA', 'regions_UNK',
    # 'countryOfOrigin_freq_enc', 'countryOfOrigin_NA', 'countryOfOrigin_AF',
    # 'countryOfOrigin_AS', 'countryOfOrigin_EU', 'countryOfOrigin_OC',
    # 'countryOfOrigin_SA', 'countryOfOrigin_UNK'
]


def apply_log_scale(df: pd.DataFrame, columns: list[str]=to_log) -> pd.DataFrame:
    """
    Applies log transformation to the specified columns in the dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    columns (list): List of column names to apply log transformation.
    
    Returns:
    pd.DataFrame: Dataframe with log-transformed columns.
    """
    result = df.copy()
    for col in columns:
        if col in df.columns:
            result[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    return result