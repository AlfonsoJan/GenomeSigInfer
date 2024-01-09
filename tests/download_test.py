#!/usr/bin/env python3
"""
Unit tests for the `GenomeSigInfer.download` module.
"""
import unittest
from pathlib import Path
from GenomeSigInfer import download
from GenomeSigInfer.errors.error import RefGenomeNotSupported


class TestDownloadRefGenomeError(unittest.TestCase):
    """
    A test case for the `Preprocessing` class in the `GenomeSigInfer.download` module.
    """

    def setUp(self):
        """
        Set up the test case by initializing necessary variables.
        """
        self.ref_genome = Path(
            "project/ref_genome"
        )  # Folder where the ref genome will be downloaded
        self.genome = "GRCh37"  # Reference Genome

    def test_download_ref_genome_error(self):
        """
        Test case to verify that RefGenomeNotSupported exception is raised when downloading an unsupported reference genome.
        """
        with self.assertRaises(RefGenomeNotSupported):
            download.download_ref_genome("", "wrong_ref_genome")

    def test_download_ref_genome(self):
        """
        Test case for downloading the reference genome.

        Checks if the reference genome file exists. If not, it calls the `download_ref_genome` function
        to download the reference genome file.
        """
        if not (self.ref_genome / self.genome).exists():
            download.download_ref_genome(self.ref_genome, self.genome)


if __name__ == "__main__":
    unittest.main()
