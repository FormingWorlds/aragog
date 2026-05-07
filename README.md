# Aragog

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation](https://github.com/FormingWorlds/aragog/actions/workflows/docs.yaml/badge.svg)](https://proteus-framework.org/aragog)
[![Tests](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml/badge.svg)](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml)

## About

Aragog is a Python package that computes the 1-D interior dynamics of rocky mantles that are solid, liquid, or mixed phase. It is mostly a pure Python version of the [SPIDER code](https://github.com/FormingWorlds/SPIDER) originally written in C albeit with some differences. Note that the atmosphere module in the original SPIDER code is now supported by a separate and more comprehensive Python package Atmodeller.

Documentation: <https://proteus-framework.org/aragog>

Source code: <https://github.com/FormingWorlds/aragog>

## Citation

If you use Aragog (or the original [SPIDER code](https://github.com/djbower/spider)) please cite:

- Bower, D.J., P. Sanan, and A.S. Wolf (2018), Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets, Phys. Earth Planet. Inter., 274, 49-62, doi: <https://doi.org/10.1016/j.pepi.2017.11.004>.

Open access versions of the publication are available:

- arXiv: <https://arxiv.org/abs/1711.07303>
- EarthArXiv: <https://eartharxiv.org/k6tgf>

## Installation

> **Note:** The standard way of installing this version of Aragog is within the PROTEUS Framework, as described in the [PROTEUS installation guide](https://proteus-framework.org/PROTEUS/installation.html#9-install-submodules-as-editable).  
    
### Quick install

The basic procedure is to install Aragog into a Python environment. For example, if you are using a Conda distribution to create and manage Python environments (e.g. [Anaconda](https://www.anaconda.com/download)), create a new environment noting that Aragog requires Python >= 3.10. Once created, make sure to activate the environment. To achieve this, terminal commands are given below, but you can also use the Anaconda Navigator (or similar GUI) to create and activate environments:

```sh
conda create -n aragog python
conda activate aragog
```

Alternatively, you can create and activate a [virtual environment](https://docs.python.org/3/library/venv.html).

Finally, install Aragog into the activated environment:

```sh
pip install fwl-aragog
```

### Developer install

Navigate to a location on your computer and obtain the source code using git:

```sh
git clone git@github.com:FormingWorlds/aragog.git aragog
cd aragog
```

Install Aragog with [pip](https://pip.pypa.io/en/stable/getting-started/), using an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`-e`) so changes in your working tree are picked up immediately:

```sh
pip install -e .
```

Pull in the optional dependency groups defined in `pyproject.toml` by listing the corresponding extras:

```sh
pip install -e ".[docs]"      # Zensical, mkdocstrings (build the docs site)
pip install -e ".[test]"      # pytest, pytest-cov, pytest-xdist, pytest-dependency
pip install -e ".[jax]"       # JAX, equinox, scikits-odes-sundials (production solver path)
```

Combine extras with comma separators, e.g. `pip install -e ".[docs,test,jax]"` for a full development install.

### Download data from the OSF repository

Aragog requires lookup table data storing thermophysics properties of the liquid and solid matter. These data are stored in an [OSF repository](https://osf.io/phsxf/) and on a [Zenodo record](https://zenodo.org/records/15728072). You can download it with the command:

```sh
aragog download all
```

The command `aragog env` will give you the path where the data have been downloaded. If you want to set up your own path, setup the environment variable `FWL_DATA` before running the download command:

```sh
export FWL_DATA=your_absolute_path/
```
