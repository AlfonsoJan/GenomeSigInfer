from pathlib import Path
from GenomeSigInfer.vcf import VCFMatrixGenerator

def test_filter_vcf_files():
    # Variables
    vcf_files = (Path("tests/files/test.vcf"),)
    genome = "GRCh37"
    # Call the function
    filtered_vcf = VCFMatrixGenerator.filter_vcf_files(vcf_files, genome)
    # Assert that the filtered_vcf DataFrame has the correct shape
    assert filtered_vcf.shape == (6, 11)
    assert filtered_vcf[4].unique() == ["SNP"]
    assert filtered_vcf[3].unique() == [genome]
    assert filtered_vcf[8].unique().tolist() == ["T", "C"]
