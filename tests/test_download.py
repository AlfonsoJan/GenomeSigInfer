import pytest
from GenomeSigInfer.download import *
from GenomeSigInfer.errors.error import *

# url = "https://getsamplefiles.com/download/tar/sample-1.tar"

def test_download_ref_genome_error():
    with pytest.raises(RefGenomeNotSUpported):
        download_ref_genome("", "wrong_ref_genome")
