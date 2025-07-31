from setuptools import setup, find_packages
import os

def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

def get_version():
    version = {}
    with open("polynomial_regression/__init__.py", "r", encoding="utf-8") as fp:
        exec(fp.read(), version)
    return version["__version__"]

setup(
    name="polynomial_regression_from_scratch",
    version=get_version(),
    author="M. Hossein Ghaemi",
    author_email="h.ghaemi.2003@gmail.com",
    description="A complete implementation of polynomial regression with regularization options, built from scratch",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/hghaemi/polynomial_regression_from_scratch.git",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "polynomial regression", 
        "machine learning", 
        "regularization",
        "gradient descent",
        "l1 regularization",
        "l2 regularization",
        "educational",
        "from scratch",
        "numpy",
        "data science"
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        "logistic_regression": ["*.txt", "*.md"],
    },
    entry_points={
        "console_scripts": [
        ],
    },
    zip_safe=False,
    test_suite="tests",
    platforms=["any"],
    license="MIT",
    cmdclass={},
    options={
        "bdist_wheel": {
            "universal": False,  
        }
    },

    python_requires=">=3.8",
    install_requires=[
    "numpy>=1.19.0",
    "matplotlib>=3.3.0",
    ],
)