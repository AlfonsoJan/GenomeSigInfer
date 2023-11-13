#!/usr/bin/env python3
"""
Visualization Module

This module provides functions for generating heatmaps
based on cosine similarity and Jensen Shannon Distance.
It includes methods for visualizing the best parameters, cosine similarity for different contexts,
and Jensen Shannon Distance for different contexts.

Functions:
    - heatmap_best_param(cosine_df: pd.DataFrame, figure_folder: Path) -> str
    - heatmap_cosine(cosine_df_list: List[pd.DataFrame], figure_folder: Path) -> str
    - heatmap_jens_shan(jens_shan_df_list: List[pd.DataFrame], figure_folder: Path) -> str
"""
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def heatmap_best_param(cosine_df: pd.DataFrame, figure_folder: Path) -> str:
    """
    Generate a heatmap of the best parameters based on cosine similarity.

    Args:
        cosine_df (pd.DataFrame): DataFrame containing cosine similarity values.
        figure_folder (Path): Path to the folder where the heatmap image will be saved.

    Returns:
        str: Image name of the saved heatmap.
    """
    data = cosine_df.values
    values = data[:, 0].astype(float)
    labels = data[:, 1]
    df = pd.DataFrame({"Cosine Similarity": values}, index=labels)
    ax = sns.heatmap(
        df,
        linewidth=0.5,
        fmt="",
        annot=values.reshape(-1, 1),
        cmap="crest",
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=1,
    )
    ax.set(ylabel="Parameters: (Init, Beta loss)")
    image_name = figure_folder / "cosine.params.similarity.png"
    plt.savefig(image_name, bbox_inches="tight", format="png", dpi=300, pad_inches=0.1)
    return image_name


def heatmap_cosine(cosine_df_list: list[pd.DataFrame], figure_folder: Path) -> str:
    """
    Generate a set of heatmaps based on cosine similarity for different contexts.

    Args:
        cosine_df_list (List[pd.DataFrame]): List of DataFrames containing cosine similarity values.
        figure_folder (Path): Path to the folder where the heatmap image will be saved.

    Returns:
        str: Image name of the saved heatmap.
    """
    _, axes = plt.subplots(1, len(cosine_df_list), figsize=(15, 15))
    for index, data_set in enumerate(cosine_df_list):
        cosine_df = data_set["data"]
        data = cosine_df.values.flatten()
        sorted_indices = np.argsort(data.astype(float))[::-1]
        sorted_values = data.astype(float)[sorted_indices]
        sorted_labels = np.array(cosine_df.columns)[sorted_indices]
        context = data_set["context"]
        df = pd.DataFrame({"Cosine Similarity": sorted_values}, index=sorted_labels)
        sns.heatmap(
            df,
            linewidth=0.5,
            fmt="",
            cmap="crest",
            ax=axes[index],
            annot=sorted_values.reshape(-1, 1),
            xticklabels=True,
            yticklabels=True,
            cbar_kws=dict(use_gridspec=False, location="top"),
            vmin=-1,
            vmax=1,
        )
        axes[index].set_title(f"Context of {context}")
    plt.savefig(
        figure_folder / "cosine.similarity.increased.context.png", bbox_inches="tight"
    )


def heatmap_jens_shan(
    jens_shan_df_list: list[pd.DataFrame], figure_folder: Path
) -> str:
    """
    Generate a set of heatmaps based on Jensen Shannon Distance for different contexts.

    Args:
        jens_shan_df_list (List[pd.DataFrame]): List of DataFrames
            containing Jensen Shannon Distance values.
        figure_folder (Path): Path to the folder where the heatmap image will be saved.

    Returns:
        str: Image name of the saved heatmap.
    """
    _, axes = plt.subplots(1, len(jens_shan_df_list), figsize=(15, 15))
    for index, data_set in enumerate(jens_shan_df_list):
        dist_df = data_set["data"]
        data = dist_df.values.flatten()
        sorted_indices = np.argsort(data.astype(float))
        sorted_values = data.astype(float)[sorted_indices]
        sorted_labels = np.array(dist_df.columns)[sorted_indices]
        context = data_set["context"]
        df = pd.DataFrame(
            {"Jensen Shannon Distance": sorted_values}, index=sorted_labels
        )
        sns.heatmap(
            df,
            linewidth=0.5,
            fmt="",
            cmap="crest_r",
            ax=axes[index],
            annot=sorted_values.reshape(-1, 1),
            xticklabels=True,
            yticklabels=True,
            cbar_kws=dict(use_gridspec=False, location="top"),
            vmin=0,
            vmax=1,
        )
        axes[index].set_title(f"Context of {context}")
    plt.savefig(
        figure_folder / "jens_shannon.dist.increased.context.png", bbox_inches="tight"
    )
