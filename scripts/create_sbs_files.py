#!/usr/bin/env python3
"""
Create mutliple SBS files. With increasing context.

The sbs.96.txt file contains all of the following the pyrimidine single nucleotide variants, N[{C > A, G, or T} or {T > A, G, or C}]N.
*4 possible starting nucleotides x 6 pyrimidine variants x 4 ending nucleotides = 96 total combinations.*

The sbs.1536.txt file contains all of the following the pyrimidine single nucleotide variants, NN[{C > A, G, or T} or {T > A, G, or C}]NN.
*16 (4x4) possible starting nucleotides x 6 pyrimidine variants x 16 (4x4) possible ending nucleotides = 1536 total combinations.*

The sbs.24576.txt file contains all of the following the pyrimidine single nucleotide variants, NNN[{C > A, G, or T} or {T > A, G, or C}]NNN.
*16 (4x4) possible starting nucleotides x 16 (4x4) nucleotides x 6 pyrimidine variants x 16 (4x4) nucleotides x 16 (4x4) possible ending dinucleotides = 24576 total combinations.*
"""
import sys
import click
from pathlib import Path
from DeepBayesMutSig import sbs


@click.command()
@click.option(
    "--project", type=click.Path(path_type=Path), default="project", prompt="The project folder name"
)
@click.option(
    "--vcf",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    prompt="The VCF file",
)
@click.option(
    "--genome",
    type=click.Choice(["GRCh37", "GRCh38"]),
    default="GRCh37",
    help="Choose genome version.",
)
@click.option("--bash", is_flag=True, help="Download using bash")
def main(project: Path, vcf: Path, genome: str, bash: bool) -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).
    """
    sbs_model = sbs.SBS(project=project, vcf=vcf, genome=genome, bash=bash)
    sbs_model.create_sbs_files()
    return 0


if __name__ == "__main__":
    sys.exit(main())
