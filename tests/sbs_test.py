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
		self.ref_genome = (
			"project/ref_genome/GRCh37"  # Folder where the ref genome will be downloaded
		)
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
		- The samples DataFrame has the shape of (24576, 1+1).
		- The count of mutations for a specific sample and mutation type matches the expected value.
		- The count of mutations for a specific sample matches the expected value.
		- The count of mutations for a specific sample and mutation type at a specific index matches the expected value.
		- The context_96 variable is equal to the generated SBS matrix.
		"""
		context_size = 7
		sbs_sampled_df = SBSMatrixGenerator.generate_single_sbs_matrix(
			folder=self.folder_sbs,
			vcf_files=self.vcf_files,
			ref_genome=self.ref_genome,
			max_context=context_size,
		)
		# Assert the samples DataFrame is not empty
		assert not sbs_sampled_df.empty
		# Assert the samples DataFrame has the shape of (24576, 2)
		n_classes = 6 * 4 ** (context_size - 1)
		assert sbs_sampled_df.shape == (n_classes, 2)
		# Assert the count of mutations for a specific sample and mutation type
		sample = "Thy-AdenoCa::PTC-88C"
		assert sbs_sampled_df.columns.to_list()[:2] == ["MutationType", sample]
		# Assert the count of mutations for a specific sample
		assert sbs_sampled_df[sample].sum() == 8
		# Number of T>G mutations taking into account reverse complement symmetry.
		ref_letter_loc = (context_size - 1) // 2 + 1
		is_thymine_subst = sbs_sampled_df['MutationType'].map(
			lambda x: x[ref_letter_loc] == 'T'
		)
		n_thymine_subst = sbs_sampled_df.loc[is_thymine_subst].sum(axis=0)
		# In total: 3 (C>G) + 2 (A>G) = 5.
		self.assertEqual(n_thymine_subst[sample], 5)

		df_thy = sbs_sampled_df.set_index('MutationType')[sample]

		# 1: chr2:96521744, T>C: TCC[T>C]CAT:
		# self.assertEqual(df_thy.loc['C[T>C]C'], 1)
		self.assertEqual(df_thy.loc['TCC[T>C]CAT'], 1)

		# 2: chr4:96525794, C>A: ATG[C>A]CCT
		self.assertEqual(df_thy.loc['ATG[C>A]CCT'], 1)

		# 3: chr2:96617125, T>C: ATC[T>C]TTC
		# 5: chr2:97817647, A>G: GAA(A>G)GAT.
		#                      : CTT(T>C)CTA.
		self.assertEqual(df_thy.loc['ATC[T>C]TTC'], 2)

		# 4: chr2:97817617, T>C: AGA[T>C]TAT
		self.assertEqual(df_thy.loc['AGA[T>C]TAT'], 1)

		# 6: chr2:97817659, C>T: CTC[C>T]ACG.
		self.assertEqual(df_thy.loc['CTC[C>T]ACG'], 1)

		# 7: chr2:97817661, C>T: CCA[C>T]GAT.
		self.assertEqual(df_thy.loc['CCA[C>T]GAT'], 1)

		# 8: chr2:97911732, A>G: ACA(A>G)TAA
		#                      : TGT(T>C)ATT.
		self.assertEqual(df_thy.loc['TTA[T>C]TGT'], 1)


if __name__ == "__main__":
	unittest.main()
