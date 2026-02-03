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

