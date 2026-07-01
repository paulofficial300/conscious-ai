from setuptools import setup, find_packages

setup(
    name="conscious-ai",
    version="0.1.0",
    description="A transformer-based architecture exploring machine consciousness through Global Workspace Theory and Higher-Order Thought modules",
    author="paulofficial300",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "jupyter>=1.0.0"],
    },
)
