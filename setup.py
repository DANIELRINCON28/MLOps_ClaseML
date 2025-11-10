"""
Setup script para el proyecto MLOps
"""
from setuptools import setup, find_packages

setup(
    name="mlops_pipeline",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.11",
    author="DANIELRINCON28",
    description="Pipeline MLOps para detección de fraude",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
