#!/usr/bin/env python3
"""
Run NMF with different combinations of initialization and beta loss and 1 signature.
And calculates `cosine similarity` from different results, in the results folder, of the scikit NMF function using different parameters.
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
def main(project: Path, sigs: int) -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).
    """
    nmf_params = nmf.NMF_Combinations(project=project, sigs=sigs)
    nmf_params.run_combinations()
    nmf_params.cosine_sim()
    return 0


if __name__ == "__main__":
    sys.exit(main())
