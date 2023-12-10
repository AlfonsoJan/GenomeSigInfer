#!/usr/bin/env python3
"""
This module defines the SBS class, which is responsible for parsing VCF files,
processing mutations, creating SBS files, and performing various operations on
genomic data.

Classes:
    - SBSMatrixGenerator: A class for creating maximum context Single Base Substitution (SBS) files.

Functions:
    - custom_chromosome_sort: Custom sorting function for sorting chromosome values.
    - generate_sbs_matrix_arg_checker: Decorator function for checking arguments before calling 'generate_sbs_matrix'.
    - generate_sbs_matrix: Initializes SBS with the specified project path.

Attributes:
    - helpers: Module containing utility functions.
    - logging: Module for logging information.
    - error: Module defining custom error classes.
    - matrix_operations: Module providing matrix operations.
    - VCFMatrixGenerator: Module for generating VCF matrices.

Author: J.A. Busker
"""
from functools import wraps
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from ..utils import helpers, logging
from ..errors import error
from ..matrix import matrix_operations
from ..vcf import VCFMatrixGenerator


def custom_chromosome_sort(value: str) -> int | float:
    """
    Custom sorting function for sorting the chromosomes.

    Args:
        value (str): The chromosome value to be sorted.

    Returns:
        int | float: The sorted value. If the value is numeric, it is returned as an integer;
                    otherwise, it is assigned a large float value.
    """
    if value.isdigit():
        return int(value)
    return float("inf")


def generate_sbs_matrix_arg_checker(func: callable) -> callable:
    """
    Decorator function for checking arguments before calling the 'generate_sbs_matrix' function.

    Args:
        func: The function to be wrapped.

    Returns:
        wrapper: The wrapped function.
    """

    @wraps(func)
    def wrapper(folder, vcf_files, ref_genome, genome):
        # Ensure folder is a Path object
        folder = Path(folder)
        # Correct type
        if not isinstance(vcf_files, tuple) and not isinstance(vcf_files, list):
            raise TypeError("Input 'vcf_files' must be a tuple or a list type.")
        # Check if the file exist
        exist_vcf_files = tuple(
            Path(vcf_file) for vcf_file in vcf_files if Path(vcf_file).exists()
        )
        if len(exist_vcf_files) < 1:
            raise FileNotFoundError(f"None of {', '.join(map(str, vcf_files))} exist!")
        # Ensure ref_genome is a Path object
        ref_genome = Path(ref_genome)
        # Check if the genome is supported
        helpers.check_supported_genome(genome)
        return func(folder, exist_vcf_files, ref_genome, genome)

    return wrapper


@generate_sbs_matrix_arg_checker
def generate_sbs_matrix(
    folder: Path,
    vcf_files: tuple[Path],
    ref_genome: Path,
    genome: helpers.MutationalSigantures.REF_GENOMES,
):
    """
    Initialize SBS with the specified project path.

    Args:
        folder (Path): Path of where the SBS will be saved.
        vcf_files (tuple[Path]): tuple of VCF files.
        ref_genome (Path): Path of the reference genome.
        genome (MutationalSigantures.REF_GENOMES): Reference genome.
    """
    # Log to the console
    logger = logging.SingletonLogger()
    logger.log_info("Creating SBS matrices!")
    # Create the folder if it does not exist yet
    folder.mkdir(parents=True, exist_ok=True)
    logger.log_info(f"Processing VCF files: {', '.join(map(str, vcf_files))}")
    # Filter the VCF files on chosen genome
    filtered_vcf = VCFMatrixGenerator.filter_vcf_files(vcf_files, genome)
    sbsmatrixgen = SBSMatrixGenerator(
        project=folder, vcf_file=filtered_vcf, genome=genome, ref_genome=ref_genome
    )
    sbsmatrixgen.parse_vcf()
    # Decrease the context stepwise
    matrix_operations.compress_matrix_stepwise(folder, sbsmatrixgen.samples_df)


class SBSMatrixGenerator:
    """
    SBSMatrixGenerator class for creating
    maximum context Single Base Substitution (SBS) files.

    Attributes:
        - project: Path object representing the project directory.
        - vcf_file: Filtered VCF files for parsing.
        - genome: String specifying the reference genome.
        - samples_df: DataFrame containing information about samples.
        - ref_genome_folder: Path object representing the reference genome folder.
        - _samples_df: Private attribute storing processed sample data.
    """

    def __init__(
        self, project: Path, vcf_file: pd.DataFrame, genome: str, ref_genome: Path
    ) -> None:
        """
        Initialize the class

        Args:
            project (Path): Path of where the SBS will be saved.
            vcf_files (pd.DataFrame): vcf_file: Filtered VCF files for parsing.
            genome (MutationalSigantures.REF_GENOMES): Reference genome.
            ref_genome (Path): Path of the reference genome.
        """
        self.project = Path(project)
        self.vcf = vcf_file
        self.genome = genome
        self.ref_genome = ref_genome
        self._logger: logging.SingletonLogger = logging.SingletonLogger()
        self._samples_df = None

    def parse_vcf(self) -> None:
        """
        Parses the VCF file and processes mutations,
        creating a max context SBS file.
        """
        # Group the VCF per chromosome
        sorted_chr, grouped_chromosomes = self.group_vcf_chromosome()
        # Init of the sampled dataframe
        self._samples_df: pd.DataFrame = matrix_operations.create_mutation_samples_df(
            self.vcf
        )
        # loop over the dataframe based on the chromosome
        for chrom_start in sorted_chr:
            indices = grouped_chromosomes[chrom_start]
            self.process_chromosome(chrom_start, indices)

    def process_chromosome(self, chrom_start: str, indices: list[int]) -> None:
        """
        Process mutations on a specific chromosome.

        Args:
            chrom_start (str): Start position of the chromosome.
            indices (List[int]): List of indices corresponding to the chromosome.
        """
        # Retrieve the content of the chromosome file as bytes
        chrom_string = self.get_chromosome_file(chrom_start)
        self._logger.log_info(f"Starting on chromosome {chrom_start}")
        # Iterate over the indices corresponding to mutations on the chromosome
        for idx in tqdm(indices):
            # Process each row in the VCF file for the current chromosome
            self.process_vcf_row(idx, chrom_string)
        self._logger.log_info(f"Finished chromosome {chrom_start}\n")

    def process_vcf_row(self, idx: int, chrom_string: bytes) -> None:
        """
        Process a single row from the VCF file.

        Args:
            idx (int): Index of the row in the VCF file.
            chrom_string (bytes): Content of the chromosome file as bytes.
        """
        # Retrieve the VCF row data at the specified index
        df_row = self.vcf.loc[idx]
        # Construct a unique identifier for the sample (e.g., "ALL::PD3954a")
        sample = f"{df_row[0]}::{df_row[1]}"
        # Adjust the mutation position to 0-based indexing
        pos = df_row[6] - 1
        # Extract mutation information
        mut = df_row[8] + ">" + df_row[9]
        # Create left and right mutation context
        left_context, right_context = self.create_mutation_context(pos, chrom_string)
        # Form the complete mutation type string
        mutation_type = f"{left_context}[{mut}]{right_context}"
        # Increment the corresponding count in the samples DataFrame
        self._samples_df.loc[
            self._samples_df["MutationType"] == mutation_type, sample
        ] += 1

    def create_mutation_context(self, pos: int, chrom_string: bytes) -> tuple[str, str]:
        """
        Create mutation context for a given position in the chromosome.
        Create the mutation in the style of eg: A[C>A]A

        Args:
            pos (int): Position of the mutation.
            chrom_string (bytes): Content of the chromosome file as bytes.

        Returns:
            Tuple[str, str]: Left and right mutation context.
        """
        # Extract the left and right context around the mutation position
        left_context = "".join(
            [
                helpers.TSB_REF[chrom_string[_]]
                for _ in range(
                    pos - helpers.MutationalSigantures.CONTEXT_LIST[0] // 2, pos
                )
            ]
        )
        right_context = "".join(
            [
                helpers.TSB_REF[chrom_string[_]]
                for _ in range(
                    pos + 1, pos + helpers.MutationalSigantures.CONTEXT_LIST[0] // 2 + 1
                )
            ]
        )
        return left_context, right_context

    def get_chromosome_file(self, chromosome: str) -> bytes:
        """
        Retrieve the content of a chromosome file.

        Args:
            chromosome (str): The chromosome number.

        Returns:
            bytes: The content of the chromosome file as bytes.

        Raises:
            FileNotFoundError: If the chromosome file does not exist.
        """
        # Construct the filename for the chromosome
        file_name = self.ref_genome / f"{chromosome}.txt"
        chrom_string = None
        try:
            # Read the content of the chromosome file as bytes
            with open(file_name, "rb") as open_file:
                chrom_string = open_file.read()
        except FileNotFoundError:
            # Raise an error if the chromosome file is not found
            raise error.RefGenomeChromosomeNotFound(chromosome) from FileNotFoundError
        return chrom_string

    def group_vcf_chromosome(self) -> tuple[list[str], dict]:
        """
        Group the VCF DataFrame by chromosome.

        Returns:
            tuple[list[str], dict]: Tuple containing
                sorted chromosomes, and grouped chromosomes.
        """
        # Group the VCF DataFrame by the CHROM column (column with index 5)
        grouped_chromosomes: dict = self.vcf.groupby(5).groups
        # Extract the chromosome names
        chromosomes: list[str] = list(grouped_chromosomes.keys())
        # Sort the chromosomes using the custom sorting function
        sorted_chr: list[str] = sorted(chromosomes, key=custom_chromosome_sort)
        return sorted_chr, grouped_chromosomes

    @property
    def samples_df(self) -> None | pd.DataFrame:
        """
        Private attribute storing processed sample data.

        Returns:
            pd.DataFrame: Max context samples dataframe.
        """
        if self._samples_df is None:
            return pd.DataFrame()
        return self._samples_df.loc[(self._samples_df != 0).any(axis=1)]
