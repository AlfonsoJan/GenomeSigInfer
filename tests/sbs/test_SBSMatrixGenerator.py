import pytest
from pathlib import Path
from GenomeSigInfer.sbs import SBSMatrixGenerator
from GenomeSigInfer import download

@pytest.fixture(scope="module")
def sbs_matrix_generator():
    ref_genome = "project/ref_genome" # Folder where the ref genome will be downloaded
    genome = "GRCh37" # Reference Genome
    if not Path(ref_genome).exists():
        download.download_ref_genome(ref_genome, genome)
    # Setup code here
    sbs_matrix_generator = SBSMatrixGenerator.generate_single_sbs_matrix(
        folder="project",
        vcf_files=["tests/files/test.vcf"],
        ref_genome=Path(ref_genome) / genome,
        genome=genome,
        max_context=3
    )
    yield sbs_matrix_generator

def test_shape_df(sbs_matrix_generator):
    # Assert the samples DataFrame has the shape of (96, 2)
    assert sbs_matrix_generator.shape == (96, 2)

def test_parse_vcf_columns(sbs_matrix_generator):
    # Assert the samples DataFrame is not empty
    assert not sbs_matrix_generator.empty
    
    
    # Assert the count of mutations for a specific sample and mutation type
    assert sbs_matrix_generator.columns.to_list() == ["MutationType", "Thy-AdenoCa::PTC-88C"]

def test_parse_vcf_count(sbs_matrix_generator):
    # Assert the samples DataFrame is not empty
    assert not sbs_matrix_generator.empty
    # Assert the count of mutations for a specific sample
    assert sbs_matrix_generator["Thy-AdenoCa::PTC-88C"].sum() == 6
    # Assert the count of mutations for a specific sample and mutation type
    assert sbs_matrix_generator["Thy-AdenoCa::PTC-88C"][34] == 1
