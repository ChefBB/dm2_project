"""
This module contains functions for imputing missing values.
"""
from typing import Callable
import pandas as pd


plaw_like = [
    # 'startYear', 'endYear', 
    'deltaYear', 'totalMedia', 'numRegions',
    'totalNominations', 'deltaCredits', 'reviewsTotal',
    'ratingCount', 'castNumber', 'companiesNumber',
    'writerCredits', # Really?
    'directorsCredits', # Really?
]


def impute_data(train: pd.DataFrame, test: pd.DataFrame | None=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Impute missing values in the training and testing datasets.

    Parameters:
        train (pd.DataFrame): The training DataFrame to impute.
        test (pd.DataFrame | None): The testing DataFrame to impute. If None, only the training DataFrame is processed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None]: The imputed training DataFrame and the imputed testing DataFrame (if provided).
    """
    # Apply imputation to the training dataset
    
    runtime_imputer = impute_runtime_minutes(train)

    train_res = train.copy()
    
    train_res['runtimeMinutes'] = runtime_imputer(train)

    
    # compute deltaYear
    train_res['deltaYear'] = train_res['endYear'] - train_res['startYear']

    # If a testing dataset is provided, apply the same imputation
    test_res = test.copy() if test is not None else None
    
    if test is not None:
        test_res['runtimeMinutes'] = runtime_imputer(test_res)
        
        test_res['deltaYear'] = test_res['endYear'] - test_res['startYear']
        
    for feat in plaw_like:
        imputer = impute_plaw_distrib_feat(train_res, feat)
        train_res[feat] = imputer(train_res)
        
        if test is not None:
            test_res[feat] = imputer(test_res)

    return train_res, test_res


def impute_runtime_minutes(df: pd.DataFrame, perc: float | None=0.99) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Impute missing values in the 'runtimeMinutes' column of the given DataFrame.
    Assigns to missing values randomly sampled data out of the central perc% range.
    Also imputes the values for rows outside the perc range if not None.
    Imputation is done separately for each 'titleType' category.

    Parameters:
        df (pd.DataFrame): The DataFrame to impute.
        
        perc (float | None): The central percentile range for imputing values. Default is 0.9.

    Returns:
        Callable[[pd.DataFrame], pd.Series]: A function that takes a DataFrame and returns the imputed 'runtimeMinutes' column.
    """
    # If perc is not None, calculate the lower and upper bounds for the central perc% range
    if perc is not None:
        lower_bound = (1 - perc) / 2
        upper_bound = 1 - lower_bound
        perc_threshold = df.groupby('titleType')['runtimeMinutes'].quantile([lower_bound, upper_bound]).unstack()

    # Define the percentiles for each titleType category, while cutting off outliers
    percentiles = df.groupby('titleType')['runtimeMinutes'].quantile([0.3, 0.7]).unstack()

    def impute_rt_mins(df: pd.DataFrame) -> pd.Series:
        """
        Impute missing values in the 'runtimeMinutes' column of the given DataFrame.
        Assigns to missing values randomly sampled data out of the central perc% range.
        Imputation is done separately for each 'titleType' category.
        Parameters:
            df (pd.DataFrame): The DataFrame to impute.
        Returns:
            pd.Series: The imputed 'runtimeMinutes' column.
        """
        # Group the data by 'titleType'
        groups = df.groupby('titleType')['runtimeMinutes']

        # Create a copy of the original column to preserve order
        imputed_runtime = df['runtimeMinutes'].copy()

        # Iterate over each group and impute missing values
        for title_type, group in groups:
            lower = percentiles.loc[title_type, 0.3]
            upper = percentiles.loc[title_type, 0.7]

            # Get valid values within the 30-70 percentile range
            valid_values = group[(group >= lower) & (group <= upper)].dropna()

            # Filter the group to include only rows within the central perc% range
            if perc is not None:
                central_lower = perc_threshold.loc[title_type, lower_bound]
                central_upper = perc_threshold.loc[title_type, upper_bound]
                valid_values = valid_values[(valid_values >= central_lower) & (valid_values <= central_upper)]

            for index in group.index:
                # If the value is outside the central perc% range, assign a random sample from the valid values
                if group[index] < lower or group[index] > upper:
                    imputed_runtime.loc[index] = valid_values.sample(n=1, replace=True, random_state=42).values[0]

            # Sample values for missing entries
            missing_count = group.isna().sum()
            if missing_count > 0:
                sampled_values = valid_values.sample(n=missing_count, replace=True, random_state=42)
                # Assign sampled values to the missing positions
                imputed_runtime.loc[group.index[group.isna()]] = sampled_values.values

        return imputed_runtime
    
    return impute_rt_mins


def impute_plaw_distrib_feat(df: pd.DataFrame, feat: str, perc: float=0.995) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Sets very high/low values to a threshold for simil-Power Law distribution.
    The imputation is done separately for each 'titleType' category.
    
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be imputed.
    feat : str
        The name of the feature/column in the DataFrame to be imputed.

    perc : float, optional
        The percentile threshold for imputing values. Default is 0.995.
        Values above this percentile will be set to the threshold value.

    Returns
    -------
    Callable[[pd.DataFrame], pd.Series]
        A function that, when applied to a DataFrame, returns a Series with the imputed values for the specified feature.
    """
    # Compute the threshold
    thresholds = df.dropna().groupby('titleType')[feat].quantile(perc)
    
    def impute_feat(df: pd.DataFrame) -> pd.Series:
        """
        Impute the specified feature in the DataFrame by setting values above a threshold to the threshold value.
        
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data to be imputed.

        Returns
        -------
        pd.Series
            A Series with the imputed values for the specified feature.
        """
        # Create a copy of the original column to preserve order
        imputed_feat = df.copy()
        
        for type in imputed_feat['titleType'].unique():
            # Get the threshold for the current titleType
            threshold = int(thresholds.loc[type])
            
            median = int(imputed_feat.loc[
                imputed_feat['titleType'] == type, feat
            ].median())
            
            # Set values above the threshold to the threshold value
            imputed_feat.loc[
                imputed_feat['titleType'] == type, feat
            ] = imputed_feat.loc[
                imputed_feat['titleType'] == type, feat
            ].clip(upper=threshold)
                
            imputed_feat.loc[
                imputed_feat['titleType'] == type, feat
            ] = imputed_feat.loc[
                imputed_feat['titleType'] == type, feat
            ].fillna(median)
        
        return imputed_feat[feat]
    
    return impute_feat