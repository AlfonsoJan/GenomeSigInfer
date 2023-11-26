#!/usr/bin/env python3
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..utils.logging import SingletonLogger
from ..utils.helpers import MutationalSigantures, MUTATION_TYPES, TSB_REF
from ..data.matrix_operations import init_mutation_df
from ..error.errors import RefGenomeChromosomeNotFound


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


class VCFGenerator:
    def __init__(self, project: Path, vcf: tuple[Path], genome: str) -> None:
        self.project = project
        self.ref_genome_folder: Path = project / "ref_genome" / genome
        self.vcf_files = vcf
        self.genome = genome
        self._logger = SingletonLogger()
        self.vcf = None
        self._samples_df = None

    @property
    def samples_df(self) -> None | pd.DataFrame:
        if self._samples_df is None: return pd.DataFrame()
        return self._samples_df.loc[(self._samples_df != 0).any(axis=1)]

    def filter_files(self) -> None:
        self._logger.log_info("Creating one large VCF matrix")
        dfs = [self.read_vcf_file(vcf_file) for vcf_file in self.vcf_files]
        self.vcf = pd.concat(dfs, ignore_index=True)

    def read_vcf_file(self, vcf_file: Path) -> pd.DataFrame:
        """
        Filter the vcf file
        """
        df = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
            df = pd.read_csv(vcf_file, sep="\t", header=None)
        mutations = np.array(df[8].astype(str) + ">" + df[9].astype(str))
        filtered_df = df[
            (df.iloc[:, 3] == self.genome)
            & (
                np.isin(mutations, MUTATION_TYPES)
                & ((df[4] == "SNP") | (df[4] == "SNV"))
            )
        ]
        filtered_df = filtered_df.astype({5: str})
        return filtered_df

    def parse_vcf(self) -> None:
        """
        Parse the VCF file and process mutations.

        Create a max context SBS file and decompress the context down.
        """
        self._logger.log_info(
            f"Processing VCF files: {', '.join(map(str, self.vcf_files))}"
        )
        sorted_chr, grouped_chromosomes = self.group_vcf_chromosome()
        samples: np.ndarray = np.array(
            self.vcf[0].astype(str) + "::" + self.vcf[1].astype(str)
        )
        self._samples_df: pd.DataFrame = init_mutation_df(
            np.unique(samples), MutationalSigantures.CONTEXT_LIST[0]
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
        df_row = self.vcf.loc[idx]
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
                TSB_REF[chrom_string[_]]
                for _ in range(pos - MutationalSigantures.CONTEXT_LIST[0] // 2, pos)
            ]
        )
        right_context = "".join(
            [
                TSB_REF[chrom_string[_]]
                for _ in range(
                    pos + 1, pos + MutationalSigantures.CONTEXT_LIST[0] // 2 + 1
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
        file_name = self.ref_genome_folder / f"{chromosome}.txt"
        chrom_string = None
        try:
            with open(file_name, "rb") as open_file:
                chrom_string = open_file.read()
        except FileNotFoundError:
            raise RefGenomeChromosomeNotFound(chromosome) from FileNotFoundError
        return chrom_string

    def group_vcf_chromosome(self) -> tuple[list[str], dict]:
        """
        Group the VCF DataFrame by chromosome.

        Returns:
            tuple[list[str], dict]: Tuple containing
                sorted chromosomes, and grouped chromosomes.
        """
        grouped_chromosomes: dict = self.vcf.groupby(5).groups
        chromosomes: list[str] = list(grouped_chromosomes.keys())
        sorted_chr: list[str] = sorted(chromosomes, key=custom_sort)
        return sorted_chr, grouped_chromosomes
