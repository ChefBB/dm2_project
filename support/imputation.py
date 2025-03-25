import pandas as pd


def impute_runtime_minutes(df: pd.DataFrame) -> pd.Series:
    """
    Impute missing values in the 'runtimeMinutes' column of the given DataFrame.
    Assigns to missing values randomly sampled data out of the 30-70 percentile range.
    Imputation is done separately for each 'titleType' category.

    Parameters:
        df (pd.DataFrame): The DataFrame to impute.

    Returns:
        pd.Series: The imputed 'runtimeMinutes' column.
    """
    # Define the percentiles for each titleType category
    percentiles = df.groupby('titleType')['runtimeMinutes'].quantile([0.3, 0.7]).unstack()

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

        # Sample values for missing entries
        missing_count = group.isna().sum()
        if missing_count > 0:
            sampled_values = valid_values.sample(n=missing_count, replace=True, random_state=42)
            # Assign sampled values to the missing positions
            imputed_runtime.loc[group.index[group.isna()]] = sampled_values.values

    return imputed_runtime