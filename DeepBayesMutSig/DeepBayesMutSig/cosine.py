#!/usr/bin/env python3
""""
This module contains functions that calculates the most similar column
based on cosine similarity for a given DataFrame containing signature data.

Functions:
- most_similarity_decompose(filename: str) -> np.ndarray: Calculate the most similar column based on cosine similarity.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


def most_similarity_decompose(filename: str) -> np.ndarray:
    """
    Calculate the most similar column based on cosine similarity.

    Args:
        filename (str): file containing signature data.

    Returns:
        np.ndarray: An array with the similarities.
    """
    dataframe = pd.read_csv(filename, sep="\t")
    label_encoder = LabelEncoder()
    df_encoded = dataframe.apply(label_encoder.fit_transform)
    cosine_similarities = cosine_similarity(
        df_encoded.T, df_encoded["SigProfiler"].values.reshape(1, -1)
    )
    col_array = np.array(dataframe.columns).reshape(-1, 1)
    sorted_indices = np.argsort(cosine_similarities.ravel().astype(float))[::-1][1:]
    values = cosine_similarities[sorted_indices]
    labels = col_array[sorted_indices]
    return np.hstack((values, labels))
