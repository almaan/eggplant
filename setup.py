#!/usr/bin/env python3

import setuptools

with open("requirements.txt") as fr:
    reqs = fr.readlines()
reqs = [r.rstrip("\n") for r in reqs]

with open("pypi-README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spatial-eggplant",
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
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.7",
    install_requires=reqs,
)
