#!/usr/bin/env python3
"""
Module for installing reference genomes.
Note:
    For now, the module only supports the GRCh37 genome.
"""
import shutil
import os
from pathlib import Path


def genome_install(
    genome: str,
    install_path: str,
    bash: bool = True,
) -> None:
    """
    Install your desired reference genome.
    (For now only GRCh37)

    Args:
        genome (str): The desired reference genome.
        install_path (str): The install location of the reference genome.
        bash (bool): If you want to install without using bash.

    Returns:
        pd.DataFrame: The df with the chances.
    """
    genomes = ["GRCh37"]
    if genome not in genomes:
        print(f"For now this only support {','.join(genomes)}")
        return
    wget_install = shutil.which("wget") is not None
    tar_install = shutil.which("tar") is not None
    if not wget_install:
        print(
            "You do not have wget installed. Please install wget and then reattempt to install."
        )
        return
    if not tar_install:
        print(
            "You do not have tar installed. Please install tar and then reattempt to install."
        )
        return
    install_path = Path(install_path) / genome
    if install_path.exists():
        shutil.rmtree(install_path)
    print(f"Creating folder '{install_path}'.")
    install_path.mkdir(parents=True, exist_ok=True)
    if (install_path / "installation.log").exists():
        os.remove(install_path / "installation.log")
    print("Beginning installation. This may take up to 40 minutes to complete.")
    cmd = f"wget -r -l1 -c -nc --no-parent -nd -P {str(install_path)} ftp://ngs.sanger.ac.uk/scratch/project/mutographs/SigProf/{genome}.tar.gz 2>> {str(install_path / 'installation.log')}"
    if bash:
        cmd = f"bash -c '{cmd}'"
    try:
        os.system(cmd)
        os.system(
            f"tar -xzf {str(install_path / genome)}.tar.gz -C {str(install_path.parent)}"
        )
        os.remove(f"{str(install_path / genome)}.tar.gz")
        print(f"Finished installing to {install_path}!")
    except FileNotFoundError as e:
        print(
            "The Sanger ftp site is not responding. Please check your internet connection/try again later."
        )
