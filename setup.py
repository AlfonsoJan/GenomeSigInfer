from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="GenomeSigInfer",
    version="1.0.0",
    packages=find_packages(),
    package_data={"": ["*.md"]},
    include_package_data=True,
    author="J.A. Busker",
    author_email="alfonsobusker@gmail.com",
    description="This project aims to refine the statistical model and the current representation of mutations in building mutational signatures in cancer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlfonsoJan/GenomeSigInfer",
    install_requires=[
        "numpy==1.23.1",
        "pandas==1.5.0",
        "pyarrow==14.0.1",
        "fastparquet==2023.10.1",
        "tqdm==4.66.1",
        "scikit_learn==1.3.1",
        "seaborn==0.13.0",
        "matplotlib==3.7.1",
        "requests==2.31.0",
        "pdoc==14.1.0"
    ],
    extras_require={
        "test": ["pytest==7.4.2", "pylint==3.0.2"],
    },
    python_requires=">=3.10",
)