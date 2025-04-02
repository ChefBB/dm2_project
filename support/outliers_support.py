"""
Some constants and functions to help with outliers detection
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from h2o.estimators.extended_isolation_forest import H2OExtendedIsolationForestEstimator
import h2o
from h2o.frame import H2OFrame
import plotly.express as px



feats_to_keep_iso_forest = [
    'startYear', 'runtimeMinutes',
    'totalCredits', 'criticReviewsTotal',
    'numRegions', 'userReviewsTotal', 'ratingCount',
    'castNumber', 'companiesNumber', 'averageRating', 'externalLinks',
    'writerCredits', 'directorsCredits', 'totalMedia',
    'totalNominations',
    'regions_freq_enc', 'regions_EU', 'regions_NA', 'regions_AS',
    'regions_AF', 'regions_OC', 'regions_SA', 'regions_UNK',
    'countryOfOrigin_freq_enc', 'countryOfOrigin_NA', 'countryOfOrigin_AF',
    'countryOfOrigin_AS', 'countryOfOrigin_EU', 'countryOfOrigin_OC',
    'countryOfOrigin_SA', 'countryOfOrigin_UNK',
    # 'endYear', 
]


def iso_forest_titletype(df: pd.DataFrame,
                         feats: list[str]=feats_to_keep_iso_forest,
                         ) -> pd.Series:
    """
    Applies Isolation Forest to detect outliers in the DataFrame.
    The function scales the data and applies Isolation Forest separately for each titleType.
    The outlier predictions are stored in a new column 'outlier'.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.
    feats : list[str], optional
        The features to use for the model. Defaults to feats_to_keep_iso_forest.
    Returns
    -------
    pd.Series: A Series with outlier predictions, maintaining the original order.
    """
    # Initialize a list to store processed groups
    processed_groups = []

    # Group by titleType and apply Isolation Forest
    for _, group in df.groupby('titleType'):
        # Preserve the original index
        original_index = group.index
        group = group.reset_index(drop=True)

        # Apply Isolation Forest
        iso_forest = (
            IsolationForest(random_state=42, contamination=0.01))
        group['outlier'] = iso_forest.fit_predict(group[feats])

        # Restore the original index
        group.index = original_index
        processed_groups.append(group)

    # Concatenate all groups and sort by the original index
    result = pd.concat(processed_groups).sort_index()

    return result['outlier']


def iso_forest_full_dataset(df: pd.DataFrame, feats: list[str]=feats_to_keep_iso_forest)->pd.Series:
    """
    Applies Isolation Forest to detect outliers in the entire DataFrame.
    The outlier predictions are stored in a new column 'outlier'.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.

    Returns
    -------
    pd.Series: A Series with outlier predictions, maintaining the original order.
    """
    # Preserve the original index
    original_index = df.index

    # Reset index for processing
    df = df.reset_index(drop=True)

    # Apply Isolation Forest
    iso_forest = IsolationForest(random_state=42, contamination=0.01)
    df['outlier'] = iso_forest.fit_predict(df[feats])

    # Restore the original index
    df.index = original_index

    return df['outlier']


def global_extended_iso_forest(df: pd.DataFrame, feats: list[str] = feats_to_keep_iso_forest) -> pd.Series:
    """
    Applies H2O Extended Isolation Forest to detect outliers in the DataFrame.
    The outlier predictions are stored in a new column 'outlier_h2o'.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.
    feats : list[str], optional
        The features to use for the model. Defaults to feats_to_keep_iso_forest.

    Returns
    -------
    pd.Series: A series with outlier predictions, maintaining the original order.
    """

    # Initialize H2O
    h2o.init()

    # Convert DataFrame to H2OFrame
    h2o_df = H2OFrame(df[feats])

    # Initialize the H2O Extended Isolation Forest model
    eif_model = H2OExtendedIsolationForestEstimator(seed=42, sample_size=256, ntrees=100)

    # Train the model
    eif_model.train(training_frame=h2o_df)

    # Predict outliers
    predictions = eif_model.predict(h2o_df)

    # Extract the anomaly score
    anomaly_scores = predictions['anomaly_score'].as_data_frame().values.flatten()

    # Determine the threshold for the top 1% of outliers
    threshold = pd.Series(anomaly_scores).quantile(0.99)

    # Assign -1 to the top 1% of outliers and 1 to the rest
    result = pd.Series((anomaly_scores >= threshold).astype(int)).replace({0: 1, 1: -1})

    # Shutdown H2O
    h2o.shutdown(prompt=False)

    return result


def classwise_extended_iso_forest(df: pd.DataFrame, feats: list[str] = feats_to_keep_iso_forest) -> pd.Series:
    """
    Applies H2O Extended Isolation Forest to detect outliers in the DataFrame.
    The outlier predictions are stored in a new column 'outlier_h2o'.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.
    feats : list[str], optional
        The features to use for the model. Defaults to feats_to_keep_iso_forest.

    Returns
    -------
    pd.Series: A Series with outlier predictions, maintaining the original order.
    """
    processed_groups = []

    # Initialize H2O
    h2o.init()

    for _, group in df.groupby('titleType'):
        # Preserve the original index
        original_index = group.index
        group = group.reset_index(drop=True)
        
        # Convert DataFrame to H2OFrame
        h2o_df = H2OFrame(group[feats])

        # Initialize the H2O Extended Isolation Forest model
        eif_model = H2OExtendedIsolationForestEstimator(seed=42, sample_size=256, ntrees=100)

        # Train the model
        eif_model.train(training_frame=h2o_df)

        # Predict outliers
        predictions = eif_model.predict(h2o_df)

        # Extract the anomaly score
        anomaly_scores = predictions['anomaly_score'].as_data_frame().values.flatten()

        # Determine the threshold for the top 1% of outliers
        threshold = pd.Series(anomaly_scores).quantile(0.99)
        
        group['outlier'] = pd.Series((anomaly_scores >= threshold).astype(int)).replace({0: 1, 1: -1})

        # Restore the original index
        group.index = original_index
        processed_groups.append(group)

    # Assign -1 to the top 1% of outliers and 1 to the rest
    result = pd.concat(processed_groups).set_index(df.index)['outlier']

    # Shutdown H2O
    h2o.shutdown(prompt=False)

    return result



def plot_3d_outliers(
    df: pd.DataFrame, title_type: str = None, feats: list[str] = feats_to_keep_iso_forest,
    outlier_col: str = 'outlier'):
    """
    Draws an interactive 3D scatter plot of the PCA components, highlighting outliers in red and others in blue.
    Allows filtering by titleType.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be plotted.
    title_type : str, optional
        The titleType to filter the data. If None, the entire dataset is used.
    feats : list[str], optional
        The features to use for PCA. Defaults to feats_to_keep_iso_forest.

    Returns
    -------
    None
    """
    # Filter by titleType if specified
    if title_type:
        df = df[df['titleType'] == title_type]

    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=3, random_state=42)
    pca_components = pca.fit_transform(df[feats])
    df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2', 'PCA3'], index=df.index)

    # Add outlier and titleType information
    df_pca['outlier'] = df[outlier_col]
    df_pca['titleType'] = df['titleType']

    # Plot using Plotly
    fig = px.scatter_3d(
        df_pca,
        x='PCA1',
        y='PCA2',
        z='PCA3',
        color=df_pca['outlier'].map({1: 'blue', -1: 'red'}),
        symbol='titleType',
        title="3D PCA Scatter Plot with Outliers Highlighted",
        labels={'color': 'Outlier', 'symbol': 'Title Type'},
        opacity=0.3,  # Correct parameter for transparency
    )
    fig.update_traces(marker_size=3)
    fig.show()