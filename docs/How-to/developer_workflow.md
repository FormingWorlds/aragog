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
