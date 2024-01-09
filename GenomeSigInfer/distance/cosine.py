#!/usr/bin/env python3
""""
This module contains functions that calculates the most similar column based on cosine similarity for a given DataFrame containing signature data.

Functions:
* cosine_nmf_w(optimal_columns: dict, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame: Calculate the cosine similarity on the most optimal columns on (decompressed) nmf result.

Author: J.A. Busker
"""
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def cosine_nmf_w(
    optimal_columns: dict, df1: pd.DataFrame, df2: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the cosine similarity on the most optimal columns on (decompressed) nmf result.

    Args:
        optimal_columns: (dict): Dictionary containing the most optimal columns.
        df1 (pd.DataFrame): DataFrame containing signature data.
        df2 (pd.DataFrame): DataFrame containing signature data.

    Returns:
        pd.DataFrame: And df with the cosine similarities
    """
    # Initialize StandardScaler to standardize features
    scaler = StandardScaler()
    # Scale the data using StandardScaler
    data1_scaled = scaler.fit_transform(df1.values)
    data1_scaled = pd.DataFrame(data1_scaled, columns=df1.columns)
    data2_scaled = scaler.transform(df2.values)
    data2_scaled = pd.DataFrame(data2_scaled, columns=df2.columns)
    # Initialize a dictionary to store cosine similarities between optimal columns
    row = {}
    # Calculate cosine similarity for each pair of optimal columns
    for col1, col2 in optimal_columns.items():
        array1 = data1_scaled[col1].values.reshape(1, -1)
        array2 = data2_scaled[col2].values.reshape(1, -1)
        similarity = cosine_similarity(array1, array2, dense_output=False)
        row[col1] = similarity[0, 0]
    return row
