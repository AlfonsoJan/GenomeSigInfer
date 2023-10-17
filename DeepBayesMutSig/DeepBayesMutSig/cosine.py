#!/usr/bin/env python3
""""
This module contains functions that calculates the most similar column
based on cosine similarity for a given DataFrame containing signature data.

Functions:
- most_similarity_decompose(filename: str) -> np.ndarray: Calculate the most similar column based on cosine similarity.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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


def cosine_mut_prob(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the cosine similarity. Of the mutation probability

    Args:
        df1 (pd.DataFrame): DataFrame containing signature data.
        df2 (pd.DataFrame): DataFrame containing signature data.

    Returns:
        pd.DataFrame: And df with the cosine similarities
    """
    scaler = StandardScaler()
    data1_scaled = scaler.fit_transform(df1.iloc[:, 1:].values)
    data2_scaled = scaler.transform(df2.iloc[:, 1:].values)
    columns = df1.columns[1:]
    result = pd.DataFrame()
    row = {}
    for column in columns:
        index = columns.get_loc(column)
        array1 = data1_scaled[:, index].reshape(1, -1)
        array2 = data2_scaled[:, index].reshape(1, -1)
        if np.linalg.norm(array1) == 0 or np.linalg.norm(array2) == 0:
            row[column] = 0.0
            continue
        similarity = cosine_similarity(array1, array2)
        row[column] = similarity[0, 0]
    result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    return result
