#!/usr/bin/env python3

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eggplant",
    version="0.1",
    author="Alma Andersson",
    author_email="almaan@kth.se",
    description="Landmark-based transfer of spatial transcriptomics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/almaan/eggplant",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Tested on Linux",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "anndata>=0.7.5",
        "matplotlib>=3.3.3",
        "scanpy>=1.5.0",
        "torch>=1.8.1",
        "gpytorch>=1.4.2",
        "squidpy>=1.0.0",
        "morphops>=0.1.12",
        "kneed>=0.7.0",
    ],
)
