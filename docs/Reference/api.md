# API overview

This is a detailed overview of aragog's API for the user's reference. If you want to understand the underlying model, please visit the [model overview](../Explanations/model.md). 

## Directory structure

```
aragog
├── cfg
│   ├── abe_liquid.cfg
│   ├── abe_mixed.cfg
│   ├── abe_mixed_init.cfg
│   ├── abe_mixed_lookup.cfg
│   ├── abe_solid.cfg
│   └── __init__.py
├── cli.py
├── core.py
├── data.py
├── __init__.py
├── interfaces.py
├── mesh.py
├── output.py
├── parser.py
├── phase.py
├── solver.py
└── utilities.py
```

Here, the `cfg` subdirectory contains config files used as input for aragog, that can be adjusted or added to. 

## High-level architecture

Aragog is structured around a small number of components:

1. **Configuration and scaling** ([`parser.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/parser.py))
2. **Mesh and equation-of-state (EOS) setup** ([`mesh.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/mesh.py))
3. **Phase/property evaluators** ([`phase.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/phase.py), [`interfaces.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/interfaces.py))
4. **State evaluation (fluxes, heating)** ([`solver.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/solver.py))
5. **Time integration** ([`solver.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/solver.py) → `scipy.integrate.solve_ivp`)
6. **Postprocessing and export** ([`output.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/output.py))

## Module reference

### aragog.cli
::: aragog.cli
    options:
      members: true
      inherited_members: true
      show_source: true

### aragog.data
::: aragog.data
    options:
      members: true
      inherited_members: true
      show_source: true

### aragog.core
::: aragog.core
    options:
      members: true
      inherited_members: true
      show_source: true

### aragog.interfaces
::: aragog.interfaces
    options:
      members: true
      inherited_members: true
      show_source: true

### aragog.mesh
::: aragog.mesh
    options:
      members: true
      inherited_members: true
      show_source: true

### aragog.output
::: aragog.output
    options:
      members: true
      inherited_members: true
      show_source: true

### aragog.parser
::: aragog.parser
    options:
      members: true
      inherited_members: true
      show_source: true

### aragog.phase
::: aragog.phase
    options:
      members: true
      inherited_members: true
      show_source: true

### aragog.solver
::: aragog.solver
    options:
      members: true
      inherited_members: true
      show_source: true

### aragog.utilities
::: aragog.utilities
    options:
      members: true
      inherited_members: true
      show_source: true

### aragog (package)
::: aragog
    options:
      members: true
      inherited_members: true
      show_source: true
