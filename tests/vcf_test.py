import unittest
from pathlib import Path
from GenomeSigInfer.vcf import VCFMatrixGenerator

class TestFilterVCFFiles(unittest.TestCase):
    def test_filter_vcf_files(self):
        # Variables
        vcf_files = (Path("tests/files/test.vcf"),)
        genome = "GRCh37"
        # Call the function
        filtered_vcf = VCFMatrixGenerator.filter_vcf_files(vcf_files, genome)
        # Assert that the filtered_vcf DataFrame has the correct shape
        self.assertEqual(filtered_vcf.shape, (6, 11))
        self.assertListEqual(filtered_vcf[4].unique().tolist(), ["SNP"])
        self.assertListEqual(filtered_vcf[3].unique().tolist(), [genome])
        self.assertListEqual(filtered_vcf[8].unique().tolist(), ["T", "C"])

if __name__ == '__main__':
    unittest.main()
