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

## CLI (`cli.py`)

The CLI defines these entry points:

### Command groups

- `cli` — top-level group
- `download` — subgroup for data download

### Commands

- `download all`
  - Calls `data.DownloadLookupTableData()` to download lookup-table data.
- `env`
  - Prints the data directory location via `data.FWL_DATA_DIR`.

## Data download (`data.py`)

### Constants

- `FWL_DATA_DIR: pathlib.Path`
  - From environment variable `FWL_DATA`, otherwise defaults to a per-user data directory via `platformdirs.user_data_dir("fwl_data")`.
- `project_id = "phsxf"` (OSF project)
- `basic_list`, `full_list`
  - Folder path tuples defining available datasets.

### Functions

#### `GetFWLData() -> Path`
Returns the absolute filesystem path to the root “FWL data” directory.

#### `DownloadLookupTableData(fname: str = "") -> None`
Downloads lookup table data into:

```
GetFWLData() / "interior_lookup_tables" / <folder>
```

Behavior:
- If `fname == ""`: downloads `basic_list`.
- If `fname` is in `full_list`: downloads only that folder.

Download strategy for each folder:
1. Try Zenodo using `zenodo_get <record_id> -o <folder_dir>`
2. If Zenodo fails: fall back to OSF via `osfclient`

### Helper functions 

- `get_zenodo_record(folder: str) -> str | None`
  - Returns a Zenodo record ID for a known folder.
- `download_zenodo_folder(folder: str, data_dir: Path) -> None`
  - Runs `zenodo_get` via subprocess, logs stdout/stderr into `zenodo.log` under `GetFWLData()`.
- `download_OSF_folder(*, storage, folders: list[str], data_dir: Path) -> None`
  - Streams matching OSF files to local disk.
