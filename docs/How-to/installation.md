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

Install Aragog into the environment with [pip](https://pip.pypa.io/en/stable/getting-started/), using an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`-e`) so changes in your working tree are picked up immediately:

```sh
pip install -e .
```

To pull in the optional dependency groups defined in `pyproject.toml`, add the corresponding extra:

```sh
pip install -e ".[docs]"      # Zensical, mkdocstrings (build the docs site)
pip install -e ".[test]"      # pytest, pytest-cov, pytest-xdist, pytest-dependency
pip install -e ".[jax]"       # JAX, equinox, scikits-odes-sundials (production solver path)
```

Combine extras with comma separators, e.g. `pip install -e ".[docs,test,jax]"` for a full development install.

## Optional dependency: SUNDIALS CVODE

`solver_method = "cvode"` is the default integrator and requires `scikits_odes` for the SUNDIALS dispatch. Without it the solver issues a warning and falls back to scipy `Radau` or `BDF`, which are sufficient for short tests but can collapse their step size at the crystallisation front on long magma-ocean cooling runs. For production-grade stiff integration, install `scikits_odes`:

```sh
pip install scikits-odes
```

`scikits_odes` is an optional dependency: importing Aragog without it succeeds, but selecting the CVODE path raises a clear error message at solve time.

## Optional dependency: JAX (production runs)

`solver.use_jax_jacobian = true` (the default) requires JAX at runtime. JAX builds the analytic Jacobian via `jax.jacrev` and feeds it to CVODE through a factory registered by the PROTEUS wrapper (or by user code that calls `EntropySolver.set_jax_cvode_factory`). Without JAX, the solver silently falls back to CVODE's finite-difference Jacobian, which is correct but slower and noisier near the rheological transition.

The simplest path is the `jax` extra defined in `pyproject.toml`:

```sh
pip install -e ".[jax]"
```

Or install the dependencies directly:

```sh
pip install jax equinox
```

`equinox` is the JAX-compatible PyTree framework used by `aragog.jax` to declare static parameter modules. The PROTEUS-side conda environment already pulls both transitively through Atmodeller; a standalone Aragog install for development without JAX is fully supported.

!!! warning "JAX is a runtime requirement for the default solver path"
    With `use_jax_jacobian = true` (default) and `solver_method = "cvode"` (default), the JAX path is exercised on every solve call. In production runs, missing JAX silently falls back to a slower finite-difference Jacobian; standalone runs that explicitly need the JAX-traced Jacobian must install the `[jax]` extra.

## Equation-of-state tables

The entropy solver requires a directory of pressure-entropy (P-S) lookup tables in the SPIDER format. The files needed and their format are documented in [Reference: data](../Reference/data.md). In coupled PROTEUS runs the wrapper produces these tables automatically from a configured P-T melting curve and the [Wolf & Bower (2018)](https://scixplorer.org/abs/2018PEPI..278...59W/abstract) RTpress liquid EOS; for standalone use, point the `eos_dir` argument of `EntropySolver.from_file()` at any directory containing the ten required files.
