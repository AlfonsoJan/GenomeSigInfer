#!/usr/bin/env python3
"""
SBS Module

This module defines the SBS class, which is responsible for parsing VCF files,
processing mutations, creating SBS files, and performing various operations on
genomic data.
"""
import sys
import os
import shutil
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from .matrix_operations import init_mutation_df, df2csv, compress
from .logging import SingletonLogger
from .errors import RefGenomeNotCorrectError
from .helpers import calculate_value, custom_sort, MUTATION_TYPES, TSB_REF


class SBS:
    """
    Parse the VCF file and process mutations.

    Create a max context SBS file and decompress the context down.
    """

    REF_GENOMES: list = ["GRCh37", "GRCh38"]
    MAX_CONTEXT: int = 7
    SORT_REGEX: dict = {
        13: r"(\w\w\w\w\w\w\[.*\]\w\w\w\w\w\w)",
        11: r"(\w\w\w\w\w\[.*\]\w\w\w\w\w)",
        9: r"(\w\w\w\w\[.*\]\w\w\w\w)",
        7: r"(\w\w\w\[.*\]\w\w\w)",
        5: r"(\w\w\[.*\]\w\w)",
        3: r"(\w\[.*\]\w)",
    }

    def __init__(
        self, project: Path, vcf: Path, genome: str, bash: bool = True
    ) -> None:
        """
        Initialize SBS with the specified project path.

        Args:
            project (Path): Path of the project.
            vcf_files (Path): List of VCF files.
            genome (str): Reference genome.
            bash (bool, optional): Use bash for installation. Defaults to True.
        """
        self._context_list: list[int] = list(range(SBS.MAX_CONTEXT, 2, -2))
        self._sizes: list[int] = [calculate_value(i) for i in self._context_list[::-1]]
        if genome not in SBS.REF_GENOMES:
            raise RefGenomeNotCorrectError(genome)
        self.genome: str = genome
        self.project: Path = project
        self._logger: SingletonLogger = SingletonLogger()
        self.vcf: Path = vcf
        self.bash: bool = bash
        self.ref_genome_folder: Path = self.project / "ref_genome" / self.genome

    def _prepare_folder(self) -> None:
        """
        Prepare main folder for SBS files.
        """
        if self.project.exists():
            shutil.rmtree(self.project)
        self._logger.log_info(f"Creating a clean project in: '{self.project}'")
        self.project.mkdir(parents=True, exist_ok=True)

    def create_sbs_files(self) -> None:
        """
        Create SBS files based on the processed VCF file.
        """
        self._prepare_folder()
        self.download_ref_genome()
        self._prepare_sbs_folder()
        self.parse_vcf()
        self._compress_one_down()

    def parse_vcf(self) -> None:
        """
        Parse the VCF file and process mutations.

        Create a max context SBS file and decompress the context down.
        """
        self._logger.log_info(f"Processing VCF file: {self.vcf}")
        filtered_df, sorted_chr, grouped_chromosomes = self.group_vcf_chromosome()
        samples: np.ndarray = np.array(
            filtered_df[0].astype(str) + "::" + filtered_df[1].astype(str)
        )
        self.samples_df: pd.DataFrame = init_mutation_df(
            np.unique(samples), self._context_list[0]
        )
        for chrom_start in sorted_chr:
            indices = grouped_chromosomes[chrom_start]
            chrom_string = None
            file_name = self.ref_genome_folder / f"{chrom_start}.txt"
            with open(file_name, "rb") as open_file:
                chrom_string = open_file.read()
            print("\n")
            self._logger.log_info(f"Starting on chromosome {chrom_start}\n")
            for idx in tqdm(indices):
                df_row = filtered_df.loc[idx]
                sample = f"{df_row[0]}::{df_row[1]}"
                pos = df_row[6] - 1
                mut = df_row[8] + ">" + df_row[9]
                # Create the mutation in the style of eg: A[C>A]A
                left_context = "".join(
                    [
                        TSB_REF[chrom_string[_]]
                        for _ in range(pos - self._context_list[0] // 2, pos)
                    ]
                )
                right_context = "".join(
                    [
                        TSB_REF[chrom_string[_]]
                        for _ in range(pos + 1, pos + self._context_list[0] // 2 + 1)
                    ]
                )
                mutation_type = f"{left_context}[{mut}]{right_context}"
                # Add 1 to the sample df
                self.samples_df.loc[
                    self.samples_df["MutationType"] == mutation_type, sample
                ] += 1
            print("\n")
            self._logger.log_info(f"Finished chromosome {chrom_start}\n")

    def _compress_one_down(self):
        """
        Compress the SBS data to lower context sizes.
        """
        sampled_one_down = pd.DataFrame()
        for context in self._context_list:
            self._logger.log_info(f"Creating a SBS matrix with context: {context}")
            if context == self.MAX_CONTEXT:
                sampled_one_down = self.samples_df
                print(sampled_one_down)
            else:
                sampled_one_down = compress(sampled_one_down, self.SORT_REGEX[context])
            filename = self.project / "SBS" / f"sbs.{sampled_one_down.shape[0]}.txt"
            df2csv(sampled_one_down, filename, sep=",")
            self._logger.log_info(f"Created a SBS matrix with context: {context}")

    def get_vcf_matrix(self) -> pd.DataFrame:
        """
        Read and filter the VCF file.
        """
        df = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
            df = pd.read_csv(self.vcf, sep="\t", header=None)
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

    def group_vcf_chromosome(self) -> tuple[pd.DataFrame, list[str], dict]:
        """
        Group the VCF DataFrame by chromosome.

        Returns:
            tuple[pd.DataFrame, list[str], dict]: Tuple containing filtered DataFrame, sorted chromosomes, and grouped chromosomes.
        """
        filtered_df: pd.DataFrame = self.get_vcf_matrix()
        grouped_chromosomes: dict = filtered_df.groupby(5).groups
        chromosomes: list[str] = list(grouped_chromosomes.keys())
        sorted_chr: list[str] = sorted(chromosomes, key=custom_sort)
        return filtered_df, sorted_chr, grouped_chromosomes

    def _prepare_sbs_folder(self) -> None:
        """
        Prepare folders for SBS files.
        """
        sbs_folder: Path = self.project / "SBS"
        if sbs_folder.exists():
            shutil.rmtree(sbs_folder)
        sbs_folder.mkdir(parents=True, exist_ok=True)

    def download_ref_genome(self) -> None:
        """
        Download the desired reference genome
        """
        self._check_programms()
        self._logger.log_info(
            f"Beginning installation of reference {self.genome}. This may take up to 40 minutes to complete."
        )
        self._download_ref_genome()

    def _download_ref_genome(self) -> None:
        """
        Download the desired reference genome.
        """
        genome_folder: Path = self.project / "ref_genome"
        genome_folder.mkdir(parents=True, exist_ok=True)
        cmd: str = (
            f"wget -r -l1 -c -nc --no-parent -nd -P {str(genome_folder)} "
            f"ftp://ngs.sanger.ac.uk/scratch/project/mutographs/SigProf/{self.genome}.tar.gz 2>> "
            f"{str(genome_folder / 'installation.log')}"
        )
        if self.bash:
            cmd = f"bash -c '{cmd}'"
        try:
            os.system(cmd)
            os.system(
                f"tar -xzf {str(self.ref_genome_folder)}.tar.gz -C {str(genome_folder)}"
            )
            os.remove(f"{str(self.ref_genome_folder)}.tar.gz")
            self._logger.log_info(
                f"Finished installing {self.genome} to {genome_folder}!"
            )
        except FileNotFoundError as _:
            self._logger.log_info(
                (
                    "The Sanger ftp site is not responding. "
                    "Please check your internet connection/try again later."
                )
            )

    def _check_programms(self) -> None:
        """
        Check if the correct programms are installed.
        """
        wget_install: bool | None = shutil.which("wget") is not None
        tar_install: bool | None = shutil.which("tar") is not None
        if not wget_install:
            self._logger.log_info(
                "You do not have wget installed. Please install wget and then reattempt to install."
            )
            sys.exit(1)
        if not tar_install:
            self._logger.log_info(
                "You do not have tar installed. Please install tar and then reattempt to install."
            )
            sys.exit(1)

    def __repr__(self) -> str:
        """
        Return a string representation of the SBS.

        Returns:
            str: String representation.
        """
        return f"SBS(project={str(self.project)})"
