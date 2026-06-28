from setuptools import setup, find_packages

setup(
    name="xai_phasenet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "seisbench>=0.1.9",
        "packaging>=20.0"
    ],
    author="Ayrat Abdullin et al.",
    description="Explainable AI framework for PhaseNet microseismic event detection",
)
