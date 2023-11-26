#!/usr/bin/env python3
"""
mutational_signatures_installation.py

This module provides functions for checking and installing necessary programs,
as well as downloading reference genomes for mutational signatures analysis.

Functions:
    - check_programs(): Checks if the required programs (wget, tar) are installed.
    - download_ref_genome(folder: Path, genome: MutationalSigantures.REF_GENOMES, bash: bool=False):
        Downloads the specified reference genome.
"""
import sys
import os
from pathlib import Path
import shutil
from .logging import SingletonLogger
from .helpers import MutationalSigantures


def check_programms() -> None:
    """
    Check if the correct programms are installed (wget, tar).
    """
    logger: SingletonLogger = SingletonLogger()
    wget_install: bool | None = shutil.which("wget") is not None
    tar_install: bool | None = shutil.which("tar") is not None
    if not wget_install:
        logger.log_info(
            "You do not have wget installed. Please install wget and then reattempt to install."
        )
        sys.exit(1)
    if not tar_install:
        logger.log_info(
            "You do not have tar installed. Please install tar and then reattempt to install."
        )
        sys.exit(1)


def download_ref_genome(
    folder: Path, genome: MutationalSigantures.REF_GENOMES, bash: bool = False
) -> None:
    """
    Download the desired reference genome.

    Args:
        folder (Path): Path of the folder.
        genome (MutationalSigantures.REF_GENOMES): Reference genome.
        bash (bool): If you want to download using bash.
    """
    # Check if the correct programms are installed
    check_programms()
    logger: SingletonLogger = SingletonLogger()
    logger.log_info(
        f"Beginning installation of reference {genome}. "
        "This may take up to 40 minutes to complete."
    )
    # Create the folder
    genome_folder: Path = folder / "ref_genome"
    ref_genome_folder: Path = genome_folder / genome
    ref_genome_folder.mkdir(parents=True, exist_ok=True)
    cmd: str = (
        f"wget -r -l1 -c -nc --no-parent -nd -P {str(genome_folder)} "
        f"ftp://ngs.sanger.ac.uk/scratch/project/mutographs/SigProf/{genome}.tar.gz 2>> "
        f"{str(genome_folder / 'installation.log')}"
    )
    if bash:
        cmd = f"bash -c '{cmd}'"
    try:
        os.system(cmd)
        os.system(f"tar -xzf {str(ref_genome_folder)}.tar.gz -C {str(genome_folder)}")
        os.remove(f"{str(ref_genome_folder)}.tar.gz")
        logger.log_info(f"Finished installing {genome} to {genome_folder}!")
    except FileNotFoundError as _:
        logger.log_info(
            (
                "The Sanger ftp site is not responding. "
                "Please check your internet connection/try again later."
            )
        )
