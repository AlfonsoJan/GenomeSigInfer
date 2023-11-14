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
from .helpers import custom_sort, MUTATION_TYPES, TSB_REF, MutationalSigantures


class SBS:
    """
    Parse the VCF file and process mutations.

    Create a max context SBS file and decompress the context down.
    """

    def __init__(
        self, project: Path, vcf: tuple[Path], genome: str, bash: bool = True
    ) -> None:
        """
        Initialize SBS with the specified project path.

        Args:
            project (Path): Path of the project.
            vcf_files (tuple[Path]): tuple of VCF files.
            genome (str): Reference genome.
        """
        if genome not in MutationalSigantures.REF_GENOMES:
            raise RefGenomeNotCorrectError(genome)
        self.genome: str = genome
        self.project: Path = project
        self._logger: SingletonLogger = SingletonLogger()
        self.vcf_list: tuple[Path] = vcf
        self.vcf = None
        self.bash: bool = bash
        self.ref_genome_folder: Path = self.project / "ref_genome" / self.genome
        self.sbs_folder: Path = self.project / "SBS"

    def create_sbs_files(self) -> None:
        """
        Create SBS files based on the processed VCF file.
        """
        self._prepare_folder()
        self.download_ref_genome()
        self.create_one_vcf_file()
        self.parse_vcf()
        self._compress_one_down()

    def _compress_one_down(self):
        """
        Compress the SBS data to lower context sizes.
        """
        sampled_one_down = pd.DataFrame()
        for context in MutationalSigantures.CONTEXT_LIST:
            self._logger.log_info(f"Creating a SBS matrix with context: {context}")
            if context == MutationalSigantures.MAX_CONTEXT:
                sampled_one_down = self.samples_df
            else:
                sampled_one_down = compress(
                    sampled_one_down, MutationalSigantures.SORT_REGEX[context]
                )
            filename = self.project / "SBS" / f"sbs.{sampled_one_down.shape[0]}.txt"
            df2csv(sampled_one_down, filename, sep=",")
            self._logger.log_info(f"Created a SBS matrix with context: {context}")

    def parse_vcf(self) -> None:
        """
        Parse the VCF file and process mutations.

        Create a max context SBS file and decompress the context down.
        """
        self._logger.log_info(
            f"Processing VCF files: {', '.join(map(str, self.vcf_list))}"
        )
        sorted_chr, grouped_chromosomes = self.group_vcf_chromosome()
        samples: np.ndarray = np.array(
            self.vcf[0].astype(str) + "::" + self.vcf[1].astype(str)
        )
        self.samples_df: pd.DataFrame = init_mutation_df(
            np.unique(samples), MutationalSigantures.CONTEXT_LIST[0]
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
                df_row = self.vcf.loc[idx]
                sample = f"{df_row[0]}::{df_row[1]}"
                pos = df_row[6] - 1
                mut = df_row[8] + ">" + df_row[9]
                # Create the mutation in the style of eg: A[C>A]A
                left_context = "".join(
                    [
                        TSB_REF[chrom_string[_]]
                        for _ in range(
                            pos - MutationalSigantures.CONTEXT_LIST[0] // 2, pos
                        )
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
                mutation_type = f"{left_context}[{mut}]{right_context}"
                # Add 1 to the sample df
                self.samples_df.loc[
                    self.samples_df["MutationType"] == mutation_type, sample
                ] += 1
            print("\n")
            self._logger.log_info(f"Finished chromosome {chrom_start}\n")

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

    def create_one_vcf_file(self):
        """
        Create one large VCF file
        """
        self._logger.log_info("Creating one large VCF file")
        dfs = [self.read_vcf_file(vcf_file) for vcf_file in self.vcf_list]
        self.vcf = pd.concat(dfs, ignore_index=True)

    def _prepare_folder(self) -> None:
        """
        Prepare main folder for SBS files.
        """
        if self.project.exists():
            shutil.rmtree(self.project)
        self._logger.log_info(f"Creating a clean project in: '{self.project}'")
        self.project.mkdir(parents=True, exist_ok=True)
        self.sbs_folder.mkdir(parents=True, exist_ok=True)

    def download_ref_genome(self) -> None:
        """
        Download the desired reference genome
        """
        self._check_programms()
        self._logger.log_info(
            f"Beginning installation of reference {self.genome}. "
            "This may take up to 40 minutes to complete."
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
        return (
            f"SBS(project={str(self.project)}, genome={self.genome}, "
            f"vcf={', '.join(map(str, self.vcf_list))}, bash={self.bash})"
        )
