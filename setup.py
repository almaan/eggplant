#!/usr/bin/env python3

from setuptools import setup, find_packages
from pathlib import Path

with open("requirements.txt") as fr:
    reqs = fr.readlines()
reqs = [r.rstrip("\n") for r in reqs]

with open("pypi-README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spatial-eggplant",
    version="0.2.3",
    author="Alma Andersson",
    author_email="almaan@kth.se",
    description="Landmark-based transfer of spatial transcriptomics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/almaan/eggplant",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.7",
    install_requires=[
        i.strip() for i in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    extras_require=dict(
        dev=["pre-commit>=2.9.0"],
    ),
)
