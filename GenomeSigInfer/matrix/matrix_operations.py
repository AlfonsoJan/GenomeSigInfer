#!/usr/bin/env python3
"""
This module provides functions for processing mutational data from VCF files and creating mutational signatures.
It includes functionality for sorting chromosomes, initializing mutation DataFrames, parsing VCF files, compressing dataframes, and creating SBS (Single Base Substitution) matrices based on context.

Functions:
* compress_to_96(df: pd.DataFrame) -> pd.DataFrame: Compress the DataFrame to 96 rows
* compress_matrix_stepwise(project: Path, samples_df: pd.DataFrame) -> None: Compress the SBS data to lower context sizes.
* compress(df: pd.DataFrame, regex_str: str) -> pd.DataFrame: Compress the dataframe down by grouping rows based on the regular pattern.
* create_mutation_samples_df(filtered_vcf: pd.DataFrame, context: int = helpers.MutationalSigantures.CONTEXT_LIST[0]) -> pd.DataFrame: Initialize the samples mutation DataFrame.
* increase_mutations(context: int) -> list[str]: Increases mutations in a given column based on a specified context.

Author: J.A. Busker
"""
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from ..utils import helpers, logging


def compress_to_96(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compress the DataFrame to 96 rows.

    Args:
        df (pd.DataFrame): The DataFrame to be compressed.

    Returns:
        pd.DataFrame: The compressed DataFrame with 96 rows.
    """
    # Check if the DataFrame is already compressed
    if df.shape[0] == 96:
        return df
    # Define columns for sorting
    col = "MutationType"
    sort_col = "sort_key"
    # Extract sorting key based on a regular expression
    df[sort_col] = df[col].str.extract(r"(\w\[.*\]\w)")
    # Sort the DataFrame based on the sorting key
    df = df.sort_values(sort_col)
    # Extract the sorted keys
    df_keys = df[sort_col].copy()
    # Drop unnecessary columns
    df = df.drop([sort_col, col], axis=1)
    # Determine the compression steps
    steps = int(df.shape[0] / 96)
    # Initialize the compressed DataFram
    compressed_cols = {col: df_keys[::steps]}
    # Compress the remaining columns
    for col in df.columns:
        chunks = [df[col].iloc[i : i + steps] for i in range(0, len(df[col]), steps)]
        chunk_sums = [chunk.sum() for chunk in chunks]
        compressed_cols[col] = chunk_sums
    compressed_df = pd.DataFrame(compressed_cols)
    # Reindex the compressed DataFrame
    return (
        compressed_df.set_index("MutationType")
        .reindex(helpers.MUTATION_LIST)
        .reset_index()
    )


def compress_matrix_stepwise(sbs_folder: Path, samples_df: pd.DataFrame) -> None:
    """
    Compress the SBS data to lower context sizes.

    Args:
        sbs_folder (Path): The sbs folder path.
        samples_df (pd.DataFrame): The max context SBS dataframe.
    """
    # Initialize the logger
    logger = logging.SingletonLogger()
    sampled_one_down = pd.DataFrame()
    # Create the output folder if it doesn't exist
    if not sbs_folder.is_dir():
        sbs_folder.mkdir(parents=True, exist_ok=True)
    # Iterate through different contexts
    for context in helpers.MutationalSigantures.CONTEXT_LIST:
        logger.log_info(f"Creating a SBS matrix with context: {context}")
        # Update the sampled DataFrame based on the context
        if context == helpers.MutationalSigantures.MAX_CONTEXT:
            sampled_one_down = samples_df
        else:
            sampled_one_down = compress(
                sampled_one_down, helpers.MutationalSigantures.SORT_REGEX[context]
            )
        # Write the compressed SBS matrix to a parquet file
        filename = sbs_folder / f"sbs.{sampled_one_down.shape[0]}.parquet"
        sampled_one_down.to_parquet(filename, compression="gzip")
        logger.log_info(
            f"Written the SBS matrix with context {context} to '{filename}'"
        )


def compress(df: pd.DataFrame, regex_str: str) -> pd.DataFrame:
    """
    Compress the dataframe down by grouping rows based on the regular pattern.

    Args:
        df (pd.DataFrame): The dataframe to be compressed.
        regex_str (str): Regular expression pattern for extracting sorting key.

    Returns:
        pd.DataFrame: The compressed DataFrame.
    """
    # Define columns for sorting
    col = "MutationType"
    sort_col = "sort_key"
    # Extract sorting key based on the provided regular expressio
    df[sort_col] = df[col].str.extract(regex_str)
    # Sort the DataFrame based on the sorting key
    df = df.sort_values(sort_col)
    # Group rows based on the sorting key and sum values
    compressed_df = df.groupby(sort_col).sum(numeric_only=True).reset_index()
    # Rename columns
    compressed_df.columns = [col] + list(compressed_df.columns[1:])
    return compressed_df


def create_mutation_samples_df(
    filtered_vcf: pd.DataFrame,
    context: int = helpers.MutationalSigantures.CONTEXT_LIST[0],
) -> pd.DataFrame:
    """
    Initialize the samples mutation DataFrame.

    Args:
        filtered_vcf (pd.DataFrame): Filtered VCF data.
        context (int, optional): The context for mutation. Defaults to the first context in the list.

    Returns:
        pd.DataFrame: The initialized mutation DataFrame.
    """
    # The array of unique sample names.
    samples: np.ndarray = np.array(
        filtered_vcf[0].astype(str) + "::" + filtered_vcf[1].astype(str)
    )
    samples = np.unique(samples)
    # Increase the mutations based on the context
    new_mut = increase_mutations(context)
    init_df = pd.DataFrame({"MutationType": new_mut})
    # Create DataFrames for each sample with zero values
    dfs_init = [init_df]
    for sample in samples:
        dfs_init.append(pd.DataFrame({sample: np.zeros(init_df.shape[0])}))
    # Concatenate DataFrames to create the samples mutation DataFrame
    samples_df = pd.concat(dfs_init, axis=1)
    return samples_df


def increase_mutations(context: int) -> list[str]:
    """
    Increases mutations in a given column based on a specified context.

    Args:
        context (int): The context for increasing mutations.

    Returns:
        list: A list of increased mutations based on the specified context.
    """
    if context < 3:
        raise ValueError("Context must be aleast 3")
    nucleotides = ["A", "C", "G", "T"]
    combinations = list(itertools.product(nucleotides, repeat=context - 3))
    # Generate new mutations based on the context and combinations
    new_mutations = [
        f"{''.join(combo[:len(combo)//2])}{mut}{''.join(combo[len(combo)//2:])}"
        for mut in helpers.MUTATION_LIST
        for combo in combinations
    ]
    return new_mutations
