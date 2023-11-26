#!/usr/bin/env python3
"""
SBS Module

This module defines the SBS class, which is responsible for parsing VCF files,
processing mutations, creating SBS files, and performing various operations on
genomic data.
"""
from pathlib import Path
from ..utils.helpers import MutationalSigantures, prepare_folder
from ..utils.logging import SingletonLogger
from ..utils.download import download_ref_genome
from ..vcf.VCFmatrixGenerator import VCFGenerator
from ..data.matrix_operations import compress_matrix_stepwise


def generate_sbs_matrix(
    project: Path,
    vcf: tuple[Path],
    genome: MutationalSigantures.REF_GENOMES,
    bash: bool,
):
    """
    Initialize SBS with the specified project path.

    Args:
        project (Path): Path of the project.
        vcf_files (tuple[Path]): tuple of VCF files.
        genome (MutationalSigantures.REF_GENOMES): Reference genome.
        bash (bool): If you want to download using bash.
    """
    logger: SingletonLogger = SingletonLogger()
    logger.log_info(f"Creating a clean project in: '{project}'")
    # Prepare the folder
    prepare_folder(folder=project)
    # Download ref genome
    download_ref_genome(folder=project, genome=genome, bash=bash)
    # Parse the VCF files
    vcf_model = VCFGenerator(project=project, vcf=vcf, genome=genome)
    vcf_model.filter_files()
    vcf_model.parse_vcf()
    compress_matrix_stepwise(project=project, samples_df=vcf_model.samples_df)
