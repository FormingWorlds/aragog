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

`python tools/paleos_table_sheet.py <table folder> [...] [--paleos-commit FOLDER=COMMIT]` prints a sheet for each set: its PALEOS commit, the md5 of each table, then one row per check with the measured value, the band, the source and a status. Run it with `src` of a checkout on `PYTHONPATH` to read with that code; otherwise it imports the installed aragog. The tool reads the commit from a `# PALEOS commit:` header line if a table has one; P-S tables have none, so give the commit with the flag.

The checks cover the density and expansivity contrast of melt and solid, the isentropic gradient in the mixture, the entropy of fusion and its trend with pressure, the gap between liquidus and solidus, the depth where the isentrope with a melt fraction of 0.4 at the surface reaches the solidus, and the temperature rows that a table copies.

| Status | Meaning |
|---|---|
| `pass`, `outside` | A comparison with a band that holds a published value of the same quantity. This is a reading, not a verdict: the bands come from other melting models. |
| `report` | A value with no reading. A published number of the same name can stand next to it for another composition, no source gives the quantity, the row is a property of the table, or the 2 steps of the mixed gradient differ by more than a factor 1.5. An isentrope that stays mixed up to the top of the range prints `> P_max`. |
| `n/a` | A table the sheet reads or a melting curve does not hold the pressure or the entropy of the check, a difference step leaves the mixed region, or a measured value is not finite. |

Grid and edges come from the loaded tables, so a new table set needs no edit.
