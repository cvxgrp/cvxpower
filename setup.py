from setuptools import setup, find_packages

setup(
    name="cvxpower",
    version="0.1.2",
    url="http://github.com/cvxgrp/cvxpower",
    packages=find_packages(),
    license='Apache',
    install_requires=[
        "cvxpy>=1.0.6",
        "tqdm",
    ],
    test_suite="cvxpower",
    description='Power Network Optimization and Simulation.',
)
