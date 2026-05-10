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

## Running on HPC clusters

Aragog uses the **serial** CVODE binding and never calls `MPI_Init` or any other MPI primitive. The solver is single-process; parallelism in PROTEUS-coupled runs is exposed through ensembles of independent runs (e.g. one SLURM job per planet, or `pytest -n auto` for the local test suite), not through MPI within a single integration.

That deliberate choice surfaces three failure modes on shared HPC modules where MPI libraries are loaded by default. None are bugs in Aragog; all are environment-level conflicts that the local stacks can introduce.

### MPI_Comm_dup before MPI_INIT

```text
*** The MPI_Comm_dup() function was called before MPI_INIT was invoked.
*** This is disallowed by the MPI standard.
*** Your MPI job will now abort.
```

Symptom: `aragog run …` aborts immediately with the message above. Cause: an OpenMPI module was loaded into the shell (often by a default `module load` block in `~/.bashrc` or by the cluster's login profile), and a transitive dependency of the Python stack (typically `mpi4py` imported lazily inside `netCDF4` or `scikits-odes-sundials`) tries to duplicate the world communicator before the user code calls `MPI_Init`. SUNDIALS' `mpi4py`-based wrapper does this on import in some builds.

Fix: unload the MPI module before launching Aragog.

```bash
module unload openmpi   # or: module purge && module load <minimal stack>
which mpirun || echo "MPI cleared"
aragog run config.toml --eos-dir "$ARAGOG_TEST_EOS_DIR"
```

For SLURM job scripts, prefer the same `module purge` + targeted load pattern at the top of the batch file; do not let the cluster's site-default `module load openmpi` propagate into the job environment. Reference for the underlying mpi4py initialisation behaviour: [mpi4py docs (3.0.1 PDF)](https://mpi4py.readthedocs.io/_/downloads/en/3.0.1/pdf/).

### Mismatched conda + system MPI

Symptom: `import netCDF4` or `import scikits_odes_sundials.cvode` raises a dynamic-loader error mentioning a missing `libmpi.so.<N>` or version mismatch. Cause: the conda environment built `netCDF4` against `libmpi` from `conda-forge`, but `LD_LIBRARY_PATH` points at the system OpenMPI install loaded by `module load`.

Fix: keep the conda env self-contained. Either install `nompi` builds (`conda install -c conda-forge "netcdf4=*=nompi*"`) or run with the conda env's own `libmpi` first on the path (`module purge` typically restores this).

### `mpiexec` / `srun` wrappers

Aragog must NOT be launched through `mpiexec`, `mpirun`, or `srun --mpi=pmi2`. Doing so spawns multiple ranks each running the same serial integration, which is at best wasteful (every rank computes the same trajectory) and at worst gives nondeterministic NetCDF output if multiple ranks try to write to the same file. The job script should call `aragog run …` (or PROTEUS's `proteus start …`) directly.

If you need ensembles, parallelise across array tasks (`#SBATCH --array=0-99`) and let each task run a single serial Aragog integration.

## Equation-of-state tables

The entropy solver requires a directory of pressure-entropy (P-S) lookup tables in the SPIDER format. The files needed and their format are documented in [Reference: data](../Reference/data.md).

In production runs (PROTEUS-coupled or standalone), Aragog uses the [PALEOS](https://scixplorer.org/abs/2026arXiv260503741A/abstract) multiphase equation of state (Attia et al. (2026)[^cite-attia2026]). PALEOS extends the entropy axis down to 0 K and uses a phase-aware blend of independent solid and liquid sub-tables joined through the lever rule on $\phi$, which gives a usable EOS down to fully solid late-stage mantle states where the original SPIDER table extrapolates. PROTEUS produces the PALEOS tables on-the-fly through Zalmoxis from a configured composition and P-T melting curve.

For SPIDER-parity reproduction or older reference runs, Aragog also accepts the Wolf & Bower (2018)[^cite-wolfbower2018] RTpress liquid EOS through the `phase_solid` / `phase_liquid` config keys.

For a standalone install, point the `eos_dir` argument of `EntropySolver.from_file()` at any directory containing the ten required files; the canonical file list and schema are documented in [Reference: data](../Reference/data.md).

[^cite-attia2026]: M. Attia, et al., *[PALEOS: Multiphase Equations of State and Mass-Radius Relations for Exoplanet Interiors](https://arxiv.org/abs/2605.03741)*, A&A (submitted), arXiv:2605.03741, 2026. [SciX](https://scixplorer.org/abs/2026arXiv260503741A/abstract).
[^cite-wolfbower2018]: A. S. Wolf, D. J. Bower, *[An equation of state for high pressure-temperature liquids (RTpress) with application to MgSiO3 melt](https://doi.org/10.1016/j.pepi.2018.02.004)*, Physics of the Earth and Planetary Interiors, 278, 59–74, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..278...59W/abstract).
