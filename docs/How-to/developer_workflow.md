# Developer workflow

## Comparing two snapshots

`tools/compare_snapshots.py` compares two NetCDF snapshots written by `aragog run`, variable by variable:

```console
python tools/compare_snapshots.py A.nc B.nc --ignore NAME[,NAME...]
```

The comparison is exact by value: NaN equals NaN, -0.0 equals 0.0, and data types and attributes are not compared.
For each variable that differs the tool prints the largest relative difference against `A.nc` over the positions where both values are finite, and the number of other differing positions (NaN or infinity against another value); string variables are reported as differing values.
It also lists each variable present in one file only.
Variables named in `--ignore` (comma-separated, repeatable) are marked `(ignored)`.

The exit code is 0 when nothing outside `--ignore` differs, 1 when something does, and 2 when a file cannot be read.
To check that a change leaves the solver trajectory unchanged, run the same configuration from both checkouts with a short end time, and pass the variables the change is meant to alter to `--ignore`.
A change to the per-call energy integrals, for example, alters only the energy diagnostics:

```console
python tools/compare_snapshots.py main.nc branch.nc \
    --ignore F_cmb,step_solver_residual_J,step_dE_F_int_J,step_dE_F_cmb_J \
    --ignore step_dE_Q_radio_J,step_dE_Q_tidal_J,step_dE_Q_radio_cons_J,step_dE_Q_tidal_cons_J
```

## Compare a P-S table set with published values

`python tools/paleos_table_sheet.py <table folder> [<table folder> ...]` prints a sheet for each set: its PALEOS commit, from a table header or from `--paleos-commit FOLDER=COMMIT`, the md5 of each table, then one row per check with the measured value, the band, the source and a status. The checks cover the density and expansivity contrast of melt and solid, the isentropic gradient in the mixture, the entropy of fusion and its trend with pressure, the gap between liquidus and solidus, the depth where the isentrope with a melt fraction of 0.4 at the surface reaches the solidus, and the temperature rows that a table copies. `pass` and `outside` compare with the band; `n/a` means the tables do not reach the pressure. An `outside` row is a reading, not a failure: the bands come from other melting models, and bands that the tool chose around a published value say so. Grid and edges come from the loaded tables, so a new table set needs no edit.
