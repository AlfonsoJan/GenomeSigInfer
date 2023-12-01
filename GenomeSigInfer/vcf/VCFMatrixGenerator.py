#!/usr/bin/env python3
"""
Module for filtering and processing VCF (Variant Call Format) files.

This module provides functions to read and filter VCF files based on specified criteria.
It includes the following functions:

1. `filter_vcf_files`: Filters multiple VCF files and combines them into a single DataFrame.
2. `read_vcf_file`: Reads and filters a single VCF file.

The module uses the pandas library for handling DataFrame operations and numpy for array manipulations.
Additionally, it utilizes a logging module for information logging.
"""
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from ..utils import helpers, logging


def filter_vcf_files(
    vcf_files: tuple[Path], genome: helpers.MutationalSigantures.REF_GENOMES
) -> pd.DataFrame:
    """
    Filters VCF files based on specified criteria.

    Args:
        vcf_files (tuple[Path]): tuple of VCF files.
        genome (MutationalSigantures.REF_GENOMES): Reference genome.

    Returns:
        pd.DataFrame: Filtered VCF data of all the files as a DataFrame.
    """
    # Reading and filtering individual VCF files
    dfs = [read_vcf_file(vcf_file, genome) for vcf_file in vcf_files]
    # Combining dataframes from individual files into one large dataframe
    filtered_vcf = pd.concat(dfs, ignore_index=True)
    logger = logging.SingletonLogger()
    logger.log_info(f"Created a large VCF containing {filtered_vcf.shape[0]} mutations")
    return filtered_vcf


def read_vcf_file(
    vcf_file: Path, genome: helpers.MutationalSigantures.REF_GENOMES
) -> pd.DataFrame:
    """
    Reads and filters a single VCF file.

    Args:
        vcf_file (Path): Path object representing the input VCF file.
        genome (MutationalSigantures.REF_GENOMES): Reference genome.

    Returns:
        pd.DataFrame: Filtered VCF data as a DataFrame.
    """
    # Reading the VCF file and handling potential warnings
    df = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
        df = pd.read_csv(vcf_file, sep="\t", header=None)
    # Extracting mutation information and filtering the dataframe
    mutations = np.array(df[8].astype(str) + ">" + df[9].astype(str))
    filtered_df = df[
        (df.iloc[:, 3] == genome)
        & (
            np.isin(mutations, helpers.MUTATION_TYPES)
            & ((df[4] == "SNP") | (df[4] == "SNV"))
        )
    ]
    # Converting the chromosome column to string type
    filtered_df = filtered_df.astype({5: str})
    return filtered_df
