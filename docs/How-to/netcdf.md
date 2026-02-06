# Inspecting aragog NetCDF output

This is a quick how-to for opening, inspecting, and plotting variables from aragog’s NetCDF snapshot files. This how-to assumes you have output `out/first_output.nc` from the [First Run tutorial](../Tutorials/firstrun.md), but is applicable to any output file. 

## 1. Quick command-line inspect

Show metadata, dimensions, and available variables:

```bash
ncdump -h out/first_output.nc
```

Print a variable (example: `temp_b`):

```bash
ncdump -v temp_b out/first_output.nc | head
```

