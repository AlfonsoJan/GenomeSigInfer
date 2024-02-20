#!/usr/bin/env python3
"""
This module defines the SBS class, which is responsible for parsing VCF files,
processing mutations, creating SBS files, and performing various operations on
genomic data.

Classes:
* SBSMatrixGenerator: A class for creating maximum context Single Base Substitution (SBS) files.

Functions:
* custom_chromosome_sort: Custom sorting function for sorting chromosome values.
* generate_sbs_matrix_arg_checker: Decorator function for checking arguments before calling 'generate_sbs_matrix'.
* generate_single_sbs_matrix: Generate a single SBS matrix.
* generate_sbs_matrix: Initializes SBS with the specified project path.

Attributes:
* helpers: Module containing utility functions.
* logging: Module for logging information.
* error: Module defining custom error classes.
* matrix_operations: Module providing matrix operations.
* VCFMatrixGenerator: Module for generating VCF matrices.

Author: J.A. Busker
"""
from operator import add
import mmap
from functools import reduce, wraps
from pathlib import Path
import pandas as pd
from ..utils import helpers, logging
from ..errors import error
from ..matrix import matrix_operations
from ..vcf import VCFMatrixGenerator


def custom_chromosome_sort(value: str) -> int | float:
	"""
	Custom sorting function for sorting the chromosomes.

	Args:
	    value (str): The chromosome value to be sorted.

	Returns:
	    int | float: The sorted value. If the value is numeric, it is returned as an integer;
	                otherwise, it is assigned a large float value.
	"""
	if value.isdigit():
		return int(value)
	return float("inf")


def read_genome_position(genome_version, chromosome_number, position, context, folder):
	genome_path = folder / genome_version / f"{chromosome_number}.txt"
	with open(genome_path, "rb") as file:
		with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
			start = max(position - context // 2, 0)
			end = position + context // 2 + 1
			mm.seek(start)
			sequence = mm.read(end - start)
	return "".join([helpers.TSB_REF[_] for _ in sequence])


def generate_sbs_matrix_arg_checker(func: callable) -> callable:
	"""
	Decorator function for checking arguments before calling the 'generate_sbs_matrix' function.

	Args:
	    func: The function to be wrapped.

	Returns:
	    wrapper: The wrapped function.
	"""

	@wraps(func)
	def wrapper(folder, vcf_files, ref_genome, **kwargs):
		# Ensure folder is a Path object
		folder = Path(folder)
		# Correct type
		if not isinstance(vcf_files, tuple) and not isinstance(vcf_files, list):
			raise TypeError("Input 'vcf_files' must be a tuple or a list type.")
		# Check if the file exist
		exist_vcf_files = tuple(
			Path(vcf_file) for vcf_file in vcf_files if Path(vcf_file).exists()
		)
		if len(exist_vcf_files) < 1:
			raise FileNotFoundError(f"None of {', '.join(map(str, vcf_files))} exist!")
		# Ensure ref_genome is a Path object
		ref_genome = Path(ref_genome)
		if func.__name__ == "generate_single_sbs_matrix":
			return func(folder, exist_vcf_files, ref_genome, **kwargs)
		return func(folder, exist_vcf_files, ref_genome)

	return wrapper


@generate_sbs_matrix_arg_checker
def generate_single_sbs_matrix(
	folder: Path,
	vcf_files: tuple[Path],
	ref_genome: Path,
	max_context: int = helpers.MutationalSigantures.MAX_CONTEXT,
) -> pd.DataFrame:
	"""
	Generate a single SBS matrix.

	Args:
	    folder (Path): Path of where the SBS will be saved.
	    vcf_files (tuple[Path]): Tuple of VCF files.
	    ref_genome (Path): Path of the reference genome(s).
	    max_context (int, optional): Maximum context. Defaults to helpers.MutationalSigantures.MAX_CONTEXT.

	Returns:
	    pd.DataFrame: Samples dataframe containing the SBS matrix.
	"""
	# Log to the console
	logger = logging.SingletonLogger()
	logger.log_info("Creating SBS matrices!")
	# Create the folder if it does not exist yet
	folder.mkdir(parents=True, exist_ok=True)
	logger.log_info(f"Processing VCF files: {', '.join(map(str, vcf_files))}")
	# Filter the VCF files on chosen genome
	filtered_vcf = VCFMatrixGenerator.filter_vcf_files(vcf_files)
	sbsmatrixgen = SBSMatrixGenerator(
		project=folder,
		vcf_file=filtered_vcf,
		ref_genome=ref_genome,
		max_context=max_context,
	)
	sbsmatrixgen.parse_vcf()
	return sbsmatrixgen.samples_df

def complement(sequence: str) -> str:
	"""Complement nucleotide sequence."""
	c_map = {"A": "T", "C": "G", "T": "A", "G": "C"}
	complement = map(lambda k: c_map[k], sequence.upper())
	return reduce(add, complement)

def reverse_complement(sequence: str) -> str:
	"""Reverse complement of nucleotide sequence."""
	reverse = sequence[::-1]
	return complement(reverse)

def apply_pyrimidine_first_convention(
	left_context: str, ref: str, alt: str, right_context: str
) -> str:
	"""Convert nucleotides on strand containing purine ref to complementary strand."""
	# Apply pyrimidine first convention.
	three_prime_context = left_context
	five_prime_context = right_context
	if ref in helpers.PURINE:
		ref = complement(ref)
		alt = complement(alt)
		three_prime_context = reverse_complement(right_context)
		five_prime_context = reverse_complement(left_context)
	mutation_class = f"{three_prime_context}[{ref}>{alt}]{five_prime_context}"
	return mutation_class

@generate_sbs_matrix_arg_checker
def generate_sbs_matrix(
	folder: Path,
	vcf_files: tuple[Path],
	ref_genome: Path,
):
	"""
	Initialize SBS with the specified project path.

	Args:
	    folder (Path): Path of where the SBS will be saved.
	    vcf_files (tuple[Path]): tuple of VCF files.
	    ref_genome (Path): Path of the reference genome(s).
	"""
	# Log to the console
	logger = logging.SingletonLogger()
	logger.log_info("Creating SBS matrices!")
	# Create the folder if it does not exist yet
	folder.mkdir(parents=True, exist_ok=True)
	logger.log_info(f"Processing VCF files: {', '.join(map(str, vcf_files))}")
	# Filter the VCF files on chosen genome
	filtered_vcf = VCFMatrixGenerator.filter_vcf_files(vcf_files)
	sbsmatrixgen = SBSMatrixGenerator(
		project=folder,
		vcf_file=filtered_vcf,
		ref_genome=ref_genome,
	)
	sbsmatrixgen.parse_vcf()
	# Decrease the context stepwise
	matrix_operations.compress_matrix_stepwise(
		folder, sbsmatrixgen.samples_df, helpers.MutationalSigantures.MAX_CONTEXT
	)


class SBSMatrixGenerator:
	"""
	SBSMatrixGenerator class for creating
	maximum context Single Base Substitution (SBS) files.

	Attributes:
	    - project: Path object representing the project directory.
	    - vcf_file: Filtered VCF files for parsing.
	    - samples_df: DataFrame containing information about samples.
	    - ref_genome_folder: Path object representing the reference genome folder.
	    - _samples_df: Private attribute storing processed sample data.
	"""

	def __init__(
		self,
		project: Path,
		vcf_file: pd.DataFrame,
		ref_genome: Path,
		max_context: int = helpers.MutationalSigantures.MAX_CONTEXT,
	) -> None:
		"""
		Initialize the class

		Args:
		    project (Path): Path of where the SBS will be saved.
		    vcf_files (pd.DataFrame): vcf_file: Filtered VCF files for parsing.
		    genome (MutationalSigantures.REF_GENOMES): Reference genome.
		    ref_genome (Path): Path of the reference genome.
		"""
		self.project = Path(project)
		self.vcf = vcf_file
		self.ref_genome = ref_genome
		self._logger: logging.SingletonLogger = logging.SingletonLogger()
		self._samples_df = None
		self.max_context = max_context
		self.context_list = list(range(self.max_context, 2, -2))

	def parse_vcf(self) -> None:
		"""
		Parses the VCF file and processes mutations,
		creating a max context SBS file.
		"""
		# Init of the sampled dataframe
		self._samples_df: pd.DataFrame = matrix_operations.create_mutation_samples_df(
			self.vcf, self.max_context
		)
		for index, row in self.vcf.iterrows():
			if index % 100000 == 0:
				progress = (index + 1) / self.vcf.shape[0] * 100
				print(f"Processed {progress:.2f}%", end="\r")
			sample = f"{row[0]}::{row[1]}"
			genome_version = row[3]
			chromosome_number = row[5]
			pos = row[6] - 1
			context_seq = read_genome_position(
				genome_version, chromosome_number, pos, self.max_context, self.ref_genome
			)
			left_context, right_context = (
				context_seq[: self.max_context // 2],
				context_seq[self.max_context // 2 + 1 :],
			)
			ref = row[8]
			alt = row[9]
			mutation_type = apply_pyrimidine_first_convention(
				left_context, ref, alt, right_context,
			)
			self._samples_df.loc[
				self._samples_df["MutationType"] == mutation_type, sample
			] += 1
		print(f"Processed {100:.2f}%", end="\r")

	@property
	def samples_df(self) -> None | pd.DataFrame:
		"""
		Private attribute storing processed sample data.

		Returns:
		    pd.DataFrame: Max context samples dataframe.
		"""
		if self._samples_df is None:
			return pd.DataFrame()
		return self._samples_df.loc[(self._samples_df != 0).any(axis=1)]

	def __repr__(self) -> str:
		"""
		Return a string representation of the SBSMatrixGenerator object.

		Returns:
		    str: String representation of the object.
		"""
		return (
			f"SBSMatrixGenerator(project={self.project}, "
			f"vcf_file={self.vcf}, "
			f"genome={self.genome}, "
			f"ref_genome={self.ref_genome})"
		)
