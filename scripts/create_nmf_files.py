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
    "--project", type=click.Path(), default="project", help="The project folder name"
)
@click.option("--sigs", type=click.INT, prompt="The number of signatures")
@click.option(
    "--cosmic",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    prompt="The cosmic file",
)
def main(project: Path, sigs: int, cosmic: Path) -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).
    """
    nmf_sbs = nmf.NMF_SBS(project, sigs, cosmic, "nndsvda", "frobenius")
    nmf_sbs.run_nmf()
    return 0


if __name__ == "__main__":
    sys.exit(main())
