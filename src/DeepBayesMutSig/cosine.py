#!/usr/bin/env python3
""""
This module contains functions that calculates the most similar column
based on cosine similarity for a given DataFrame containing signature data.

Functions:
- most_similarity_decompose(filename: str) -> np.ndarray: Calculate the most similar column based on cosine similarity.
"""
from ast import literal_eval
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


def most_similarity_decompose(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the most similar column based on cosine similarity.

    Args:
        dataframe (pd.DataFrame): Dataframe containing the data.

    Returns:
        pd.DataFrame: Dataframe with the similarities.
    """
    label_encoder = LabelEncoder()
    for col in dataframe.columns:
        dataframe[col] = dataframe[col].apply(literal_eval)
        dataframe[col] = dataframe[col].apply(lambda x: " ".join(x))
    df_encoded = dataframe.apply(label_encoder.fit_transform)
    cosine_similarities = cosine_similarity(
        df_encoded.T, df_encoded["sigprofiler"].values.reshape(1, -1)
    )
    col_array = np.array(dataframe.columns).reshape(-1, 1)
    sorted_indices = np.argsort(cosine_similarities.ravel().astype(float))[::-1][1:]
    values = cosine_similarities[sorted_indices]
    labels = np.array(
        [" ".join(name[0].split("_")) for name in col_array[sorted_indices]]
    )
    return pd.DataFrame({"cosine similarity": values.flatten(), "parameter": labels})


def cosine_nmf_w(
    optimal_columns: dict, df1: pd.DataFrame, df2: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the cosine similarity on the most optimal columns on (decompressed) nmf result.

    Args:
        df1 (pd.DataFrame): DataFrame containing signature data.
        df2 (pd.DataFrame): DataFrame containing signature data.

    Returns:
        pd.DataFrame: And df with the cosine similarities
    """
    scaler = StandardScaler()
    data1_scaled = scaler.fit_transform(df1.values)
    data1_scaled = pd.DataFrame(data1_scaled, columns=df1.columns)

    data2_scaled = scaler.transform(df2.values)
    data2_scaled = pd.DataFrame(data2_scaled, columns=df2.columns)
    row = {}
    for col1, col2 in optimal_columns.items():
        array1 = data1_scaled[col1].values.reshape(1, -1)
        array2 = data2_scaled[col2].values.reshape(1, -1)
        similarity = cosine_similarity(array1, array2, dense_output=False)
        row[col1] = similarity[0, 0]
    return row
