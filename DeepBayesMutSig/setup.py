from setuptools import setup, find_packages

setup(
    name="DeepBayesMutSig",
    version="0.1",
    packages=find_packages(),
    author="J.A. Busker",
    install_requires=[
        'numpy==1.23.1',
        'pandas==1.4.3',
        'scikit_learn==1.3.1',
        'SigProfilerAssignment==0.0.32',
        'SigProfilerExtractor==1.1.22',
        'seaborn==0.13.0',
        'scipy==1.11.3',
        'tqdm==4.66.1',
    ],
    extras_require={
        'test': ['pytest==7.4.2'],
    },
)
