from setuptools import setup, find_packages

setup(
    name="nn-neuron-activations",
    version="0.1.0",
    author="Mick Kalle Mickelborg",
    author_email="kallemickelborg@gmail.com",
    description="A project for analyzing neuron activations using Agentic AI concepts.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kallemickelborg/nn-neuron-activations",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "tensorflow",
        "matplotlib",
        "seaborn",
        "transformers",
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
