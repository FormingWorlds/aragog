# Installation

!!! note
    The standard way of installing this version of Aragog is within the PROTEUS Framework, as described in the [PROTEUS installation guide](https://proteus-framework.org/PROTEUS/installation.html#9-install-submodules-as-editable).  
    
## Quick install

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

## Developer install

Navigate to a location on your computer and obtain the source code using git:

```sh
git clone git@github.com:FormingWorlds/aragog.git aragog
cd aragog
```

Install Aragog into the environment using either: (a) [Poetry](https://python-poetry.org), or (b) [pip](https://pip.pypa.io/en/stable/getting-started/). 

There are some subtle differences between Poetry and pip, but in general Aragog is configured to be interoperable for most common operations (e.g. see this [Gist](https://gist.github.com/djbower/e9538e7eb5ed3deaf3c4de9dea41ebcd)). 

(a) Poetry option, which requires that [Poetry](https://python-poetry.org) is installed:

```sh
poetry install --with docs
```

(b) pip option, where the ``-e`` option is for an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html):

```sh 
pip install -e ".[docs]"
```

If desired, you will need to manually install the dependencies for the tests, which are automatically installed by Poetry but not by pip. See the additional dependencies to install in `pyproject.toml`.

More comprehensive set up guides are available here:

- [VS Code and Poetry guide](https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3)
- [Windows and Spyder guide](https://gist.github.com/djbower/c82b4a70a3c3c74ad26dc572edefdd34)

## Optional dependency: SUNDIALS CVODE

For production-grade stiff integration, install `scikits_odes` so the solver can dispatch to SUNDIALS CVODE (`solver_method = "cvode"`). Without it the solver falls back to scipy `Radau` or `BDF`, which are sufficient for short tests but can collapse their step size at the crystallisation front on long magma-ocean cooling runs:

```sh
pip install scikits-odes
```

`scikits_odes` is an optional dependency: importing Aragog without it succeeds, but selecting the CVODE path raises a clear error message at solve time.

## Equation-of-state tables

The entropy solver requires a directory of pressure-entropy (P-S) lookup tables in the SPIDER format. The files needed and their format are documented in [Reference: data](../Reference/data.md). In coupled PROTEUS runs the wrapper produces these tables automatically from a configured P-T melting curve and the Wolf-Bower (2018) RTpress liquid EOS; for standalone use, point the `eos_dir` argument of `EntropySolver.from_file()` at any directory containing the ten required files.
