#!/usr/bin/env python3
"""
Create nmf files and deconmpose mutational signatures from NMF results.
And calculates the `cosine similarity` between each file's signature data and a reference column.
"""
import sys
import click
from pathlib import Path
from DeepBayesMutSig import nmf


@click.command()
@click.option(
    "--project", type=click.Path(path_type=Path), default="project", prompt="The project folder name"
)
@click.option("--sigs", type=click.INT, prompt="The number of signatures")
@click.option(
    "--cosmic",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    prompt="The cosmic file",
)
@click.option(
    "--nmf-init",
    type=click.Choice(["random", "nndsvd", "nndsvda", "nndsvdar", "custom"]),
    default="nndsvda",
    help="NMF initialization method. Choose from 'random', 'nndsvd', 'nndsvda', 'nndsvdar', or 'custom'.",
)
@click.option(
    "--beta-loss",
    type=click.Choice(['frobenius', 'kullback-leibler', 'itakura-saito']),
    default='frobenius',
    help="Beta loss function for NMF. Choose from 'frobenius', 'kullback-leibler', or 'itakura-saito'.",
)
def main(project: Path, sigs: int, cosmic: Path, nmf_init: str, beta_loss: str) -> int:
    """
    Main entry point of the script.
    
    Args:
        project (Path): Path of the project folder.
        sigs (int): Number of signatures.
        cosmic (Path): Path of the cosmic file.
        nmf_init (str): NMF initialization method.
        beta_loss (str): Beta loss function for NMF.

    Returns:
        int: Exit status (0 for success).
    """
    nmf_sbs = nmf.NMF_SBS(project, sigs, cosmic, nmf_init, beta_loss)
    nmf_sbs.run_nmf()
    return 0


if __name__ == "__main__":
    sys.exit(main())
