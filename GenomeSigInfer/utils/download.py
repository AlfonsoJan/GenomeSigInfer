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
from . import logging, helpers


def check_programms() -> None:
    """
    Check if the correct programms are installed (wget, tar).
    """
    logger: logging.SingletonLogger = logging.SingletonLogger()
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
    folder: Path, genome: helpers.MutationalSigantures.REF_GENOMES, bash: bool = False
) -> Path:
    """
    Download the desired reference genome.

    Args:
        folder (Path): Path of the folder.
        genome (MutationalSigantures.REF_GENOMES): Reference genome.
        bash (bool): If you want to download using bash.
    """
    # Create the folder if it does not exist yet
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tar_file = folder / genome
    helpers.check_supported_genome(genome)
    # Check if the correct programms are installed
    check_programms()
    logger: logging.SingletonLogger = logging.SingletonLogger()
    logger.log_info(
        f"Beginning installation of reference {genome}. "
        "This may take up to 40 minutes to complete."
    )
    cmd: str = (
        f"wget -r -l1 -c -nc --no-parent -nd -P {str(folder)} "
        f"ftp://ngs.sanger.ac.uk/scratch/project/mutographs/SigProf/{genome}.tar.gz 2>> "
        f"{str(folder / 'installation.log')}"
    )
    if bash:
        cmd = f"bash -c '{cmd}'"
    try:
        os.system(cmd)
        os.system(f"tar -xzf {str(tar_file)}.tar.gz -C {str(folder)}")
        os.remove(f"{str(tar_file)}.tar.gz")
        logger.log_info(f"Finished installing {genome} to {folder}!")
    except FileNotFoundError as _:
        logger.log_info(
            (
                "The Sanger ftp site is not responding. "
                "Please check your internet connection/try again later."
            )
        )
    # return ref_genome_folder
