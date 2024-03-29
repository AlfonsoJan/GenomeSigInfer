[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://osf.io/t6j7u/wiki/home/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# GenomeSigInfer

GenomeSigInfer is a powerful Python framework designed for the efficient creation of mutational matrices from somatic mutations. The tool excels at identifying and categorizing mutations, specifically focusing on Single Nucleotide Variants (SNVs) and Single Nucleotide Polymorphisms (SNPs). Its primary functionality lies in the ability to unveil mutational signatures within Single Base Substitution (SBS) data, leveraging advanced techniques such as Non-Negative Matrix Factorization (NMF). Beyond signature extraction, GenomeSigInfer goes a step further, offering functionality to calculate key metrics like `cosine similarities` and `Jensen Shannon divergence` on the decompressed context sizes. And offers a seamless visualization experience. With the ability to create detailed plots for each identified mutational signature, researchers can effortlessly explore the proportion of mutations associated with specific signatures.

Detailed documentation can be found at: [docs](https://alfonsojan.github.io/GenomeSigInfer/GenomeSigInfer.html)

- [Installation](#installation)
- [Functions](#functions)
  - [Create SBS Files](#create-sbs-files)
  - [Create NMF Files](#create-nmf-files)
  - [Create Signature Plots](#create-signature-plots)

## Installation

1. Clone the repository & Install dependencies:

```bash
$ git clone https://github.com/AlfonsoJan/GenomeSigInfer
$ cd GenomeSigInfer
$ pip install .
```

2. Install your desired reference genome as follows:

```python
from GenomeSigInfer import download
ref_genome = "project/ref_genome" # Folder where the ref genome will be downloaded
genome = "GRCh37" # Reference Genome
download.download_ref_genome(ref_genome, genome)
```

## Functions

The list of available functions are:

- SBSMatrixGenerator.generate_sbs_matrix
- NMFMatrixGenerator.generate_nmf_matrix
- signature_plots.SigPlots

### Create SBS Files

Create mutliple SBS files. With increasing context.

The sbs.96.txt file contains all of the following the pyrimidine single nucleotide variants, N[{C > A, G, or T} or {T > A, G, or C}]N.
*4 possible starting nucleotides x 6 pyrimidine variants x 4 ending nucleotides = 96 total combinations.*

The sbs.1536.txt file contains all of the following the pyrimidine single nucleotide variants, NN[{C > A, G, or T} or {T > A, G, or C}]NN.
*16 (4x4) possible starting nucleotides x 6 pyrimidine variants x 16 (4x4) possible ending nucleotides = 1536 total combinations.*

The sbs.24576.txt file contains all of the following the pyrimidine single nucleotide variants, NNN[{C > A, G, or T} or {T > A, G, or C}]NNN.
*64 (4x4x4) nucleotides x 6 pyrimidine variants x 64 (4x4x4) possible ending dinucleotides = 24576 total combinations.*

```python
from GenomeSigInfer.sbs import SBSMatrixGenerator
genome = "GRCh37" # Ref genome
folder_ref_genome = "project/ref_genome/GRCh37" # The folder of the GRCh37 files
folder_sbs = "project/SBS" # Folder where the SBS files will be saved to
vcf_files = ["data/vcf/file1.vcf", "data/vcf/file2.vcf"] # List of the VCF files
SBSMatrixGenerator.generate_sbs_matrix(
  folder_sbs,
  vcf_files,
  folder_ref_genome,
  genome
)
```

### Create NMF Files

Create nmf files and deconmpose mutational signatures from NMF results. And calculate the similarities between each file's signature data and the cosmic columns.

```python
from GenomeSigInfer.nmf import NMFMatrixGenerator
sbs_folder = "project/SBS/" # Folder where the SBS files are locate
cosmic_path = "data/COSMIC_v3.4_SBS_GRCh37.txt" # Cosmic file name for the results
signatures = 48 # Amount of mutational signatures
nmf_init = "None" # init for the NMF
beta_loss = "frobenius" # beta_loss for the NMF
nmf_folder = "project/NMF/" # Folder where the NMF files are saved to
result_folder = "project/results/" # Folder where results from the analysis will be saved to
nmf_matrixgenerator = NMFMatrixGenerator.generate_nmf_matrix(
  sbs_folder,
  signatures,
  cosmic_path,
  nmf_init,
  beta_loss,
  nmf_folder,
  result_folder
)
nmf_matrixgenerator.run_nmf_on_sbs_files()

# Create images of the cosine similarities and Jensen Shannon divergence
# The figure folder
figure_folder = "../figures/"
nmf_matrixgenerator.create_distance_figures(figure_folder)
```

### Create Signature Plots

Create signature plots for all the decomposed signatures files.

```python
from GenomeSigInfer.figures import signature_plots
nmf_folder = "project/NMF" # Folder where the NMF files are located
result_folder = "project/results" # Folder where the plots are saved to
sig_plots = signature_plots.SigPlots(nmf_folder, figure_folder)
sig_plots.create_plots()
# Create plots of all the context for this signature
sig_plots.create_expected_plots(["SBS7a"])
```
