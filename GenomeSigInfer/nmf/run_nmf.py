#!/usr/bin/env python3
"""
This module defines the RunNMF class, which is responsible for running Non-Negative Matrix Factorization (NMF) on genomic data. It uses the scikit-learn library for NMF implementation.

Attributes:
* genomes (np.ndarray): Genomic data for NMF.
* signatures (int): Number of signatures to extract.
* init (str): Initialization method for NMF.
* beta_loss (str): Beta loss function for NMF.

Attributes:
* W (np.ndarray): NMF factorization matrix.

Methods:
* fit(): Fit NMF to the genomic data.

Author: J.A. Busker
"""
import numpy as np
from sklearn.decomposition import NMF


class RunNMF:
	"""
	A class for running Non-Negative Matrix Factorization (NMF) on genomic data.

	Attributes:
	    genomes (np.ndarray): Genomic data for NMF.
	    signatures (int): Number of signatures to extract.
	    init (str): Initialization method for NMF.
	    beta_loss (str): Beta loss function for NMF.

	Attributes:
	    W (np.ndarray): NMF factorization matrix.

	Methods:
	    fit(): Fit NMF to the genomic data.
	"""

	def __init__(
		self,
		genomes: np.ndarray,
		signatures: int = 1,
		init: str = "nndsvdar",
		beta_loss: str = "frobenius",
	) -> None:
		"""
		Initialize the RunNMF class.

		Args:
		    genomes (np.ndarray): Genomic data for NMF.
		    signatures (int): Number of signatures to extract.
		    init (str): Initialization method for NMF.
		    beta_loss (str): Beta loss function for NMF.
		"""
		self._solver: str = "cd" if beta_loss == "frobenius" else "mu"
		self._genomes: np.ndarray = genomes
		self._init: str | None = None if init == "None" else init
		self._beta_loss: str = beta_loss
		self._signatures: int = signatures
		self._W: np.ndarray = None
		self._Err: np.ndarray = None

	def fit(self):
		"""
		Fit NMF to the genomic data.

		This method performs NMF factorization on the genomic data.
		"""
		# Run NMF model
		# And fit the model
		nmf = NMF(
			n_components=self._signatures,
			init=self._init,
			beta_loss=self._beta_loss,
			solver=self._solver,
			max_iter=10_000,
			tol=1e-15,
		)
		self._W: np.ndarray = nmf.fit_transform(self._genomes)
		self._Err: np.ndarray = nmf.reconstruction_err_

	@property
	def W(self) -> np.ndarray:
		"""
		self._W (np.ndarray): NMF factorization matrix.
		"""
		return self._W

	@property
	def reconstruction_err(self) -> np.ndarray:
		"""
		self._Err (np.ndarray): Reconstruction Error.
		"""
		return self._Err

	@property
	def W_norm(self) -> np.ndarray:
		"""
		self._W (np.ndarray): NMF factorization matrix normalized between 0 and 1.
		"""
		return self._W / self._W.sum(axis=0)[np.newaxis]

	def __repr__(self) -> str:
		"""
		Return a string representation of the RunNMF object.

		Returns:
		    str: String representation of the object.
		"""
		return f"RunNMF(genomes={self._genomes}, signatures={self._signatures}, init={self._init}, beta_loss={self._beta_loss})"
