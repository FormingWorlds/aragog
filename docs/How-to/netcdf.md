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

## 2. Python option A: `xarray` 

Install once:

```bash
pip install xarray
```

Open and inspect:

```python
import xarray as xr

ds = xr.open_dataset("out/first_output.nc")
print(ds)          # dimensions, variables, attrs
print(ds.attrs)    # global metadata
print(ds.data_vars)
```

List variable names:

```python
print(list(ds.data_vars))
```

Grab a variable as a NumPy array:

```python
temp = ds["temp_b"].values
```

## 3. Plot a variable vs radius/pressure (xarray + matplotlib)

### Temperature vs pressure

```python
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("out/first_output.nc")

P = ds["pres_b"].values     # GPa
T = ds["temp_b"].values     # K

plt.figure()
plt.plot(T, P)
plt.gca().invert_yaxis()
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (GPa)")
plt.title("Temperature profile")
plt.tight_layout()
plt.show()
```

### Total heat flux vs radius

```python
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("out/first_output.nc")

r = ds["radius_b"].values    # km
F = ds["Ftotal_b"].values    # W m-2

plt.figure()
plt.plot(F, r)
plt.xlabel("Total heat flux (W m$^{-2}$)")
plt.ylabel("Radius (km)")
plt.title("Total heat flux profile")
plt.tight_layout()
plt.show()
```

## 4. Python option B: `netCDF4` (no extra dependencies needed)

```python
import netCDF4 as nc

ds = nc.Dataset("out/first_output.nc")

print(ds.dimensions)
print(ds.variables.keys())

temp_b = ds.variables["temp_b"][:]  # read variable
print(temp_b)

ds.close()
```
