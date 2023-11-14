#!/usr/bin/env python3
"""
Create signature plots for all the decomposed signatures files.
"""
import sys
import click
from pathlib import Path
from DeepBayesMutSig import signature_plots


@click.command()
@click.option(
    "--project", type=click.Path(path_type=Path), default="project", prompt="The project folder name"
)
def main(project: Path) -> int:
    """
    Main entry point of the script.
    
    Args:
        project (Path): Path of the project folder.

    Returns:
        int: Exit status (0 for success).
    """
    sig_plots = signature_plots.SigPlots(project)
    sig_plots.create_plots()
    return 0


if __name__ == "__main__":
    sys.exit(main())
