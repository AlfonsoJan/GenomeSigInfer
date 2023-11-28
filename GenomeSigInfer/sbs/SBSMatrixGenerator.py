#!/usr/bin/env python3
"""
SBS Module

This module defines the SBS class, which is responsible for parsing VCF files,
processing mutations, creating SBS files, and performing various operations on
genomic data.
"""
import warnings
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from ..utils import logging, helpers
from ..data import matrix_operations
from ..errors import error


def custom_sort(value: str) -> int | float:
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


def generate_sbs_matrix(
    project: Path,
    vcf: tuple[Path],
    ref_genome: Path,
    genome: helpers.MutationalSigantures.REF_GENOMES,
):
    """
    Initialize SBS with the specified project path.

    Args:
        project (Path): Path of where the SBS will be saved.
        ref_genome (Path): Path of the reference genome.
        vcf_files (tuple[Path]): tuple of VCF files.
        genome (MutationalSigantures.REF_GENOMES): Reference genome.
        bash (bool): If you want to download using bash.
    """
    project = Path(project)
    ref_genome = Path(ref_genome)
    vcf = tuple(Path(vcf_file) for vcf_file in vcf)
    logger = logging.SingletonLogger()
    logger.log_info("Creating SBS matrices!")
    # Prepare the folder
    helpers.prepare_folder_new_proj(project)
    helpers.check_file_existence(vcf)
    sbsmatrixgen = SBSMatrixGenerator(
        project=project, vcf_files=vcf, genome=genome, ref_genome=ref_genome
    )
    # Parse the VCF files
    sbsmatrixgen.filter_files()
    sbsmatrixgen.parse_vcf()
    matrix_operations.compress_matrix_stepwise(
        sbs_folder=project, samples_df=sbsmatrixgen.samples_df
    )


class SBSMatrixGenerator:
    """
    SBSMatrixGenerator class for creating
    maximum context Single Base Substitution (SBS) files.

    Attributes:
        - project: Path object representing the project directory.
        - vcf_files: Tuple of Path objects representing input VCF files.
        - genome: String specifying the reference genome.
        - samples_df: DataFrame containing information about samples.
        - ref_genome_folder: Path object representing the reference genome folder.
        - _filtered_vcf: DataFrame containing the filtered VCF data.
        - _samples_df: Private attribute storing processed sample data.
    """

    def __init__(
        self, project: Path, vcf_files: tuple[Path], genome: str, ref_genome: Path
    ) -> None:
        """
        Initialize the class

        Args:
            project (Path): Path of where the SBS will be saved.
            vcf_files (tuple[Path]): tuple of VCF files.
            genome (MutationalSigantures.REF_GENOMES): Reference genome.
            ref_genome (Path): Path of the reference genome.
        """
        self.project = Path(project)
        self.vcf_files = tuple(Path(vcf_file) for vcf_file in vcf_files)
        self.genome = genome
        self.ref_genome = ref_genome
        self._logger: logging.SingletonLogger = logging.SingletonLogger()
        self._filtered_vcf = None
        self._samples_df = None

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

    def parse_vcf(self) -> None:
        """
        Parses the VCF file and processes mutations,
        creating a max context SBS file.
        """
        self._logger.log_info(
            f"Processing VCF files: {', '.join(map(str, self.vcf_files))}"
        )
        sorted_chr, grouped_chromosomes = self.group_vcf_chromosome()
        samples: np.ndarray = np.array(
            self._filtered_vcf[0].astype(str) + "::" + self._filtered_vcf[1].astype(str)
        )
        self._samples_df: pd.DataFrame = matrix_operations.init_mutation_df(
            np.unique(samples), helpers.MutationalSigantures.CONTEXT_LIST[0]
        )
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
        chrom_string = self.get_chromosome_file(chrom_start)
        self._logger.log_info(f"Starting on chromosome {chrom_start}")
        for idx in tqdm(indices):
            self.process_vcf_row(idx, chrom_string)
        self._logger.log_info(f"Finished chromosome {chrom_start}\n")

    def process_vcf_row(self, idx: int, chrom_string: bytes) -> None:
        """
        Process a single row from the VCF file.

        Args:
            idx (int): Index of the row in the VCF file.
            chrom_string (bytes): Content of the chromosome file as bytes.
        """
        df_row = self._filtered_vcf.loc[idx]
        sample = f"{df_row[0]}::{df_row[1]}"
        pos = df_row[6] - 1
        mut = df_row[8] + ">" + df_row[9]
        left_context, right_context = self.create_mutation_context(pos, chrom_string)
        mutation_type = f"{left_context}[{mut}]{right_context}"
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
        file_name = self.ref_genome / self.genome / f"{chromosome}.txt"
        chrom_string = None
        try:
            with open(file_name, "rb") as open_file:
                chrom_string = open_file.read()
        except FileNotFoundError:
            raise error.RefGenomeChromosomeNotFound(chromosome) from FileNotFoundError
        return chrom_string

    def group_vcf_chromosome(self) -> tuple[list[str], dict]:
        """
        Group the VCF DataFrame by chromosome.

        Returns:
            tuple[list[str], dict]: Tuple containing
                sorted chromosomes, and grouped chromosomes.
        """
        grouped_chromosomes: dict = self._filtered_vcf.groupby(5).groups
        chromosomes: list[str] = list(grouped_chromosomes.keys())
        sorted_chr: list[str] = sorted(chromosomes, key=custom_sort)
        return sorted_chr, grouped_chromosomes

    def filter_files(self) -> None:
        """
        Filters VCF files based on specified criteria.
        """
        self._logger.log_info("Creating one large VCF matrix")
        dfs = [self.read_vcf_file(vcf_file) for vcf_file in self.vcf_files]
        self._filtered_vcf = pd.concat(dfs, ignore_index=True)
        self._logger.log_info(
            f"Created a large VCF containing {self._filtered_vcf.shape[0]} mutations"
        )

    def read_vcf_file(self, vcf_file: Path) -> pd.DataFrame:
        """
        Reads and filters a single VCF file.

        Args:
            vcf_file (Path): Path object representing the input VCF file.

        Returns:
            pd.DataFrame: Filtered VCF data as a DataFrame.
        """
        df = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
            df = pd.read_csv(vcf_file, sep="\t", header=None)
        mutations = np.array(df[8].astype(str) + ">" + df[9].astype(str))
        filtered_df = df[
            (df.iloc[:, 3] == self.genome)
            & (
                np.isin(mutations, helpers.MUTATION_TYPES)
                & ((df[4] == "SNP") | (df[4] == "SNV"))
            )
        ]
        filtered_df = filtered_df.astype({5: str})
        return filtered_df
