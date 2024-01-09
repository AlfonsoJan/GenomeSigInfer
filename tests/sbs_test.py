#!/usr/bin/env python3
"""
Unit tests for the `GenomeSigInfer.sbs.SBSMatrixGenerator` module.
"""
import unittest
from pathlib import Path
import pandas as pd
from GenomeSigInfer.sbs import SBSMatrixGenerator
from GenomeSigInfer import download


class TestGenerateSBSMatrix(unittest.TestCase):
    """
    A test case for the `generate_single_sbs_matrix` fucntion in the `GenomeSigInfer.sbs.SBSMatrixGenerator` module.
    """

    def setUp(self):
        """
        Set up the test environment by initializing necessary variables and downloading the reference genome if needed.
        """
        self.context_96 = pd.read_csv("tests/files/concat.sbs.96.txt", sep=",")
        self.ref_genome = "project/ref_genome/GRCh37"  # Folder where the ref genome will be downloaded
        self.genome = "GRCh37"  # Reference Genome
        self.folder_sbs = "project/SBS"
        self.vcf_files = ["tests/files/test.vcf"]
        if not Path(self.ref_genome).exists():
            download.download_ref_genome(self.ref_genome, self.genome)

    def test_generate_single_sbs_matrix(self):
        """
        Test case for generating a single SBS matrix.

        This test case verifies the correctness of the generated SBS matrix by performing the following assertions:
        - The samples DataFrame is not empty.
        - The samples DataFrame has the shape of (96, 4919).
        - The count of mutations for a specific sample and mutation type matches the expected value.
        - The count of mutations for a specific sample matches the expected value.
        - The count of mutations for a specific sample and mutation type at a specific index matches the expected value.
        - The context_96 variable is equal to the generated SBS matrix.
        """
        sbs_sampled_df = SBSMatrixGenerator.generate_single_sbs_matrix(
            folder=self.folder_sbs,
            vcf_files=self.vcf_files,
            ref_genome=self.ref_genome,
            genome=self.genome,
            max_context=3,
        )
        # Assert the samples DataFrame is not empty
        assert not sbs_sampled_df.empty
        # Assert the samples DataFrame has the shape of (96, 4919)
        assert sbs_sampled_df.shape == (96, 4919)
        # Assert the count of mutations for a specific sample and mutation type
        assert sbs_sampled_df.columns.to_list()[:2] == ["MutationType", "ALL::PD3952a"]
        # Assert the count of mutations for a specific sample
        assert sbs_sampled_df["Thy-AdenoCa::PTC-88C"].sum() == 484
        # Assert the count of mutations for a specific sample and mutation type
        assert sbs_sampled_df["Thy-AdenoCa::PTC-88C"][34] == 21
        assert self.context_96 == sbs_sampled_df


if __name__ == "__main__":
    unittest.main()
