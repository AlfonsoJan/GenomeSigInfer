import unittest
from pathlib import Path
from GenomeSigInfer import download
from GenomeSigInfer.errors.error import RefGenomeNotSUpported


class TestDownloadRefGenomeError(unittest.TestCase):
    def setUp(self):
        self.ref_genome = Path("project/ref_genome")  # Folder where the ref genome will be downloaded
        self.genome = "GRCh37"  # Reference Genome

    def test_download_ref_genome_error(self):
        with self.assertRaises(RefGenomeNotSUpported):
            download.download_ref_genome("", "wrong_ref_genome")

    def test_download_ref_genome(self):
        if not (self.ref_genome / self.genome).exists():
            download.download_ref_genome(self.ref_genome, self.genome)

if __name__ == '__main__':
    unittest.main()
