from setuptools import setup, find_packages

setup(
    name = "dem",
    version = "0.1.1",
    url = "http://github.com/cvxgrp/dem",
    packages = find_packages(),
    install_requires=[
        "cvxpy",
        "tqdm",
    ],
    test_suite = "dem",
)
