# Developer workflow

## Comparing two snapshots

`tools/compare_snapshots.py` compares two NetCDF snapshots written by `aragog run`, variable by variable:

```console
python tools/compare_snapshots.py A.nc B.nc --ignore NAME[,NAME...]
```

The comparison is bitwise, with NaN equal to NaN.
For each variable that differs the tool prints the largest relative difference against `A.nc` and the number of positions where only one file holds NaN; string variables are reported as differing values.
It also lists each variable present in one file only.
Variables named in `--ignore` (comma-separated, repeatable) are marked `(ignored)`.

The exit code is 0 when nothing outside `--ignore` differs, 1 when something does, and 2 when a file cannot be read.
To check that a change leaves the solver trajectory unchanged, run the same configuration from both checkouts with a short end time, and pass the variables the change is meant to alter to `--ignore`.
