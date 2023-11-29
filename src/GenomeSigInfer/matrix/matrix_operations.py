#!/usr/bin/env python3
"""
This module provides functions for processing mutational data
from VCF files and creating mutational signatures.
It includes functionality for sorting chromosomes,
initializing mutation DataFrames, parsing VCF files,
compressing dataframes, and creating
SBS (Single Base Substitution) matrices based on context.
Functions:
    - df2csv(df: pd.DataFrame, fname: str, formats: list[str] = [], sep: str = "\t") -> None:
        Write a DataFrame to a CSV file using a custom format.
    - compress_matrix_stepwise(project: Path, samples_df: pd.DataFrame) -> None:
        Compress the SBS data to lower context sizes.
    - compress(df: pd.DataFrame, regex_str: str) -> pd.DataFrame:
        Compress the dataframe down by grouping rows based on the regular pattern.
    - create_mutation_samples_df(filtered_vcf: pd.DataFrame) -> pd.DataFrame:
        Initialize the samples mutation DataFrame.
    - increase_mutations(context: int) -> list[str]:
        Increases mutations in a given column based on a specified context.
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
    if df.shape[0] == 96:
        return df
    col = "MutationType"
    sort_col = "sort_key"
    df[sort_col] = df[col].str.extract(r"(\w\[.*\]\w)")
    df = df.sort_values(sort_col)
    df_keys = df[sort_col].copy()
    df = df.drop([sort_col, col], axis=1)
    steps = int(df.shape[0] / 96)

    compressed_df = pd.DataFrame()
    compressed_df[col] = df_keys[::steps]
    for col in df.columns:
        chunks = [df[col][i : i + steps] for i in range(0, len(df[col]), steps)]
        chunk_sums = [chunk.sum() for chunk in chunks]
        compressed_df[col] = chunk_sums
    return compressed_df.set_index("MutationType").reindex(helpers.MUTATION_LIST).reset_index()

def df2csv(
    df: pd.DataFrame, fname: Path, formats: list[str] = [], sep: str = "\t"
) -> None:
    """
    Write a DataFrame to a CSV file using a custom format.

    Args:
        df (pd.DataFrame): The DataFrame to be written to the CSV file.
        fname (str): The filename for the CSV file.
        formats (list[str]): List of format strings for each column.
        sep (str): The separator for the CSV file.
    """
    # function is faster than to_csv
    # Only for creating SBS matrices
    if len(df.columns) <= 0:
        return
    Nd = len(df.columns)
    Nd_1 = Nd
    Nf = 0
    formats.append("%s")
    if Nf < Nd:
        for ii in range(Nf, Nd, 1):
            coltype = df[df.columns[ii]].dtype
            ff = "%s"
            if coltype == np.int64 or coltype == np.float64:
                ff = "%d"
            formats.append(ff)
    header = list(df.columns)
    with open(fname, "w", buffering=200000) as fh:
        fh.write(sep.join(header) + "\n")
        for row in df.itertuples(index=True):
            ss = ""
            for ii in range(1, Nd + 1, 1):
                ss += formats[ii] % row[ii]
                if ii < Nd_1:
                    ss += sep
            fh.write(ss + "\n")

def compress_matrix_stepwise(sbs_folder: Path, samples_df: pd.DataFrame) -> None:
    """
    Compress the SBS data to lower context sizes.

    Args:
        sbs_folder (Path): The sbs folder path.
        samples_df (pd.DataFrame): The max context SBS dataframe.
    """
    logger = logging.SingletonLogger()
    sampled_one_down = pd.DataFrame()
    if not sbs_folder.is_dir():
        sbs_folder.mkdir(parents=True, exist_ok=True)
    for context in helpers.MutationalSigantures.CONTEXT_LIST:
        logger.log_info(f"Creating a SBS matrix with context: {context}")
        if context == helpers.MutationalSigantures.MAX_CONTEXT:
            sampled_one_down = samples_df
        else:
            sampled_one_down = compress(
                sampled_one_down, helpers.MutationalSigantures.SORT_REGEX[context]
            )
        filename = sbs_folder / f"sbs.{sampled_one_down.shape[0]}.txt"
        df2csv(sampled_one_down, filename, sep=",")
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
    col = "MutationType"
    sort_col = "sort_key"
    df[sort_col] = df[col].str.extract(regex_str)
    df = df.sort_values(sort_col)
    compressed_df = df.groupby(sort_col).sum().reset_index()
    compressed_df.columns = [col] + list(compressed_df.columns[1:])
    return compressed_df

def create_mutation_samples_df(filtered_vcf: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize the samples mutation DataFrame.

    Args:
        filtered_vcf (pd.DataFrame): Filtered VCF data.

    Returns:
        pd.DataFrame: The initialized mutation DataFrame.
    """
    # The array of unique sample names.
    samples: np.ndarray = np.array(
        filtered_vcf[0].astype(str) + "::" + filtered_vcf[1].astype(str)
    )
    samples = np.unique(samples)
    # Increase the mutations based on the context
    new_mut = increase_mutations(helpers.MutationalSigantures.CONTEXT_LIST[0])
    init_df = pd.DataFrame({"MutationType": new_mut})
    # Create DataFrames for each sample with zero values
    dfs_init = [init_df]
    for sample in samples:
        dfs_init.append(pd.DataFrame({sample: np.zeros(init_df.shape[0])}))
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
    new_mutations = [
        f"{''.join(combo[:len(combo)//2])}{mut}{''.join(combo[len(combo)//2:])}"
        for mut in helpers.MUTATION_LIST
        for combo in combinations
    ]
    return new_mutations