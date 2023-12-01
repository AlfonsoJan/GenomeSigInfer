#!/usr/bin/env python3
"""
The module includes methods for downloading and extracting reference genomes.

Functions:
    - download_ref_genome_arg_checker(func: callable) -> callable:
        A decorator function to check arguments before calling the actual download_ref_genome function.
    - download_ref_genome(folder: Path, genome: MutationalSigantures.REF_GENOMES, bash: bool=False) -> None:
        Downloads the specified reference genome.
    - download_tar_url(url: str, download_path: Path, extracted_path: Path, genome: str) -> None:
        Download a tar.gz file from the provided URL, extract its contents, and clean up.
"""
import sys
import os
import requests
import tarfile
from functools import wraps
from pathlib import Path
import shutil
from .utils import helpers, logging


def download_ref_genome_arg_checker(func: callable) -> callable:
    """
    A decorator function to check arguments before calling the actual download_ref_genome function.

    Args:
        func (callable): download_ref_genome function.

    Returns:
        wrapper (callable): function with the correct type arguments.
    """

    @wraps(func)
    def wrapper(folder, genome, *args, **kwargs):
        folder = Path(folder)
        helpers.check_supported_genome(genome)
        return func(folder, genome, *args, **kwargs)

    return wrapper


@download_ref_genome_arg_checker
def download_ref_genome(
    folder: Path, genome: helpers.MutationalSigantures.REF_GENOMES
) -> None:
    """
    Download the desired reference genome.

    Args:
        folder (Path): Path of the folder.
        genome (MutationalSigantures.REF_GENOMES): Reference genome.
    """
    # Create the folder if it does not exist yet
    folder.mkdir(parents=True, exist_ok=True)
    # Name for the saved file
    download_path = folder / f"{genome}.tar.gz"
    # Log to the console
    logger: logging.SingletonLogger = logging.SingletonLogger()
    logger.log_info(
        f"Beginning downloading of reference {genome}. "
        "This may take up to 40 minutes to complete."
    )
    # Download the tar.gz file
    url = f"https://ngs.sanger.ac.uk/scratch/project/mutographs/SigProf/{genome}.tar.gz"
    download_tar_url(url, download_path, folder)


def download_tar_url(
    url: str, download_path: Path, extracted_path: Path, genome: str
) -> None:
    """
    Download a tar.gz file from the provided URL, extract its contents, and clean up.

    Args:
        url (str): URL of the tar.gz file.
        download_path (Path): Path to save the downloaded tar.gz file.
        extracted_path (Path): Path to extract the contents of the tar.gz file.
        genome (str): Name of the reference genome.
    """
    logger: logging.SingletonLogger = logging.SingletonLogger()
    # Download the tar.gz file
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        logger.log_info("Finished downloading the file")
        # Save the downloaded tar.gz file
        with open(download_path, "wb") as file:
            file.write(response.content)
        # Extract the contents of the tar.gz file
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(extracted_path)
        logger.log_info(f"Finished downloading {genome} to {extracted_path}!")
        # Clean up by removing the downloaded tar.gz file
        os.remove(download_path)
    else:
        logger.log_warning(
            (
                "The Sanger ftp site is not responding. "
                "Please check your internet connection/try again later."
            )
        )
