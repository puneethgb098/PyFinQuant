from setuptools import setup, find_packages

setup(
    name="pyfinquant",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "pytest",
        "pytest-cov"
    ],
    python_requires=">=3.8",
    author="PyFinQuant Team",
    author_email="",
    description="A Python library for quantitative finance",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
) 