from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="GenomeSigInfer",
    version="0.4.2",
    packages=find_packages(),
    author="J.A. Busker",
    author_email="alfonsobusker@gmail.com",
    description="This project aims to refine the statistical model and the current representation of mutations in building mutational signatures in cancer using deep Bayesian neural nets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlfonsoJan/GenomeSigInfer",
    install_requires=[
        "numpy==1.23.1",
        "pandas==1.4.3",
        "tqdm==4.66.1",
        "scikit_learn==1.3.1",
        "SigProfilerAssignment==0.0.32",
        "SigProfilerExtractor==1.1.22",
        "seaborn==0.13.0",
        "matplotlib==3.7.1"
    ],
    python_requires=">=3.10",
)