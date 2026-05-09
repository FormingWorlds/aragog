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

### System dependencies

The `[jax]` extra pulls `scikits-odes-sundials`, which is a thin Python binding around the SUNDIALS C library. The wheel does not bundle SUNDIALS, so the development headers must be available on the system before `pip install` runs. Pick the recipe that matches your platform:

```sh
# Debian / Ubuntu (libsundials-dev lives in the universe repo;
# enable it first if your image is stripped down, e.g. CI runners)
sudo add-apt-repository universe
sudo apt install libsundials-dev

# Fedora
sudo dnf install sundials sundials-devel

# macOS (Homebrew)
brew install sundials

# Conda / Mamba (any OS, recommended for PROTEUS)
conda install -c conda-forge sundials
```

The Fedora line installs only the serial library and headers. The `sundials-openmpi-devel` and `sundials-mpich-devel` packages are not required: Aragog uses the serial CVODE binding and never opens an MPI handle.

If `pip install -e ".[jax]"` later fails with a missing `sundials.h` or a linker error on `-lsundials_cvode`, this is the step that was skipped. The PROTEUS conda environment already provides SUNDIALS through `conda-forge`, so a PROTEUS install does not need the manual step.

### Editable install

Install Aragog into the environment with [pip](https://pip.pypa.io/en/stable/getting-started/), using an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`-e`) so changes in your working tree are picked up immediately. The `[jax]` extra is the recommended install: it pulls SUNDIALS CVODE (`scikits-odes-sundials`), JAX, and `equinox`, which together are the production solver path.

```sh
pip install -e ".[jax]"
```

To add the documentation and test toolchains, combine extras:

```sh
pip install -e ".[docs]"          # Zensical, mkdocstrings (build the docs site)
pip install -e ".[test]"          # pytest, pytest-cov, pytest-xdist, pytest-dependency
pip install -e ".[docs,test,jax]" # full development install
```

A bare `pip install -e .` is supported only for inspection of the source tree; running the solver from a bare install falls back to the scipy `Radau` / `BDF` integrators with a finite-difference Jacobian, which is correct on short tests but slow and step-size-fragile on multi-Myr cooling runs. For any actual run, install the `[jax]` extra.

## Production solver path: SUNDIALS CVODE + JAX analytic Jacobian

The production path that PROTEUS exercises on every coupling step is `solver_method = "cvode"` with `solver.use_jax_jacobian = true` (both Aragog defaults). Two runtime dependencies make this work:

- `scikits-odes-sundials` provides the SUNDIALS CVODE binding for the BDF integrator. Aragog imports CVODE directly from `scikits_odes_sundials.cvode` (it does not pull the full `scikits-odes` metapackage).
- `jax` plus `equinox` build the analytic Jacobian via `jax.jacrev` and feed it to CVODE through a factory registered by the PROTEUS wrapper (or by user code that calls `EntropySolver.set_jax_cvode_factory`). `equinox` is the JAX-compatible PyTree framework used by `aragog.jax` to declare static parameter modules.

Both are installed automatically by the `[jax]` extra. To install them by hand:

```sh
pip install scikits-odes-sundials jax equinox
```

The PROTEUS-side conda environment already pulls all three transitively through Atmodeller, so a PROTEUS install does not require this step.

!!! info "Fallbacks for development without CVODE or JAX"
    If `scikits-odes-sundials` is not importable, Aragog issues a warning and falls back to scipy `Radau` or `BDF`. If `jax` is not importable with `use_jax_jacobian = true`, it falls back to CVODE's finite-difference Jacobian, which is slower and noisier near the rheological transition. Both fallbacks are sufficient for short standalone tests but are not suitable for production-tolerance multi-Myr runs.

## Equation-of-state tables

The entropy solver requires a directory of pressure-entropy (P-S) lookup tables in the SPIDER format. The files needed and their format are documented in [Reference: data](../Reference/data.md).

In production runs (PROTEUS-coupled or standalone), Aragog uses the [PALEOS](https://scixplorer.org/abs/2026arXiv260503741A/abstract) multiphase equation of state (Attia et al. (2026)[^cite-attia2026]). PALEOS extends the entropy axis down to 0 K and uses a phase-aware blend of independent solid and liquid sub-tables joined through the lever rule on $\phi$, which gives a usable EOS down to fully solid late-stage mantle states where the original SPIDER table extrapolates. PROTEUS produces the PALEOS tables on-the-fly through Zalmoxis from a configured composition and P-T melting curve.

For SPIDER-parity reproduction or older reference runs, Aragog also accepts the Wolf & Bower (2018)[^cite-wolfbower2018] RTpress liquid EOS through the `phase_solid` / `phase_liquid` config keys.

For a standalone install, point the `eos_dir` argument of `EntropySolver.from_file()` at any directory containing the ten required files; the canonical file list and schema are documented in [Reference: data](../Reference/data.md).

[^cite-attia2026]: M. Attia, et al., *[PALEOS: Multiphase Equations of State and Mass-Radius Relations for Exoplanet Interiors](https://arxiv.org/abs/2605.03741)*, A&A (submitted), arXiv:2605.03741, 2026. [SciX](https://scixplorer.org/abs/2026arXiv260503741A/abstract).
[^cite-wolfbower2018]: A. S. Wolf, D. J. Bower, *[An equation of state for high pressure-temperature liquids (RTpress) with application to MgSiO3 melt](https://doi.org/10.1016/j.pepi.2018.02.004)*, Physics of the Earth and Planetary Interiors, 278, 59–74, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..278...59W/abstract).
