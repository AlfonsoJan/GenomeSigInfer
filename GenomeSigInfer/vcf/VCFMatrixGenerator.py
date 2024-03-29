#!/usr/bin/env python3
"""
Module for filtering and processing VCF (Variant Call Format) files.

This module provides functions to read and filter VCF files based on specified criteria.
It includes the following functions:

The module uses the pandas library for handling DataFrame operations and numpy for array manipulations.
Additionally, it utilizes a logging module for information logging.

Author: J.A. Busker
"""
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from ..utils import helpers, logging


def filter_vcf_files(vcf_files: tuple[Path]) -> pd.DataFrame:
	"""
	Filters VCF files based on specified criteria.

	Args:
	    vcf_files (tuple[Path]): tuple of VCF files.

	Returns:
	    pd.DataFrame: Filtered VCF data of all the files as a DataFrame.
	"""
	# Reading and filtering individual VCF files
	dfs = [read_vcf_file(vcf_file) for vcf_file in vcf_files]
	# Combining dataframes from individual files into one large dataframe
	filtered_vcf = pd.concat(dfs, ignore_index=True)
	logger = logging.SingletonLogger()
	logger.log_info(f"Created a large VCF containing {filtered_vcf.shape[0]} mutations")
	return filtered_vcf


def read_vcf_file(vcf_file: Path) -> pd.DataFrame:
	"""
	Reads and filters a single VCF file.

	Args:
	    vcf_file (Path): Path object representing the input VCF file.

	Returns:
	    pd.DataFrame: Filtered VCF data as a DataFrame.
	"""
	# Reading the VCF file and handling potential warnings
	df = None
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
		df = pd.read_csv(vcf_file, sep="\t", header=None)
	# Extracting mutation information and filtering the dataframe
	filtered_df = df[
		((df.iloc[:, 3] == "GRCh37") | (df.iloc[:, 3] == "GRCh38"))
		& (
			((df[4] == "SNP") | (df[4] == "SNV"))
		)
	]
	# Converting the chromosome column to string type
	filtered_df = filtered_df.astype({5: str})
	return filtered_df
