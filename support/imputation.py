"""
This module contains functions for imputing missing values.
"""
from typing import Callable
import pandas as pd


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
    train = impute_runtime_minutes(train)()

    # If a testing dataset is provided, apply the same imputation
    if test is not None:
        test = impute_runtime_minutes(test)()

    return train, test


def impute_runtime_minutes(df: pd.DataFrame, perc: float | None=0.9) -> Callable[[pd.DataFrame], pd.Series]:
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