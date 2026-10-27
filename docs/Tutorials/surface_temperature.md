# Fixed surface temperature

In this tutorial you hold the surface of a mantle shell at a fixed temperature and let the mantle below it cool. You then compare the surface heat flux and the temperature profile from Aragog with the textbook solution for a cooling half space.

## What you will learn

- How to set a prescribed surface temperature with `outer_boundary_condition = 5`.
- How to read the surface heat flux and the temperature profile from the solver output.
- How to check a run against a closed-form solution.

## What you need

- Aragog installed (see [Installation](../How-to/installation.md)).
- No equation-of-state (EOS) tables: the tutorial uses the constant-properties mode, where the temperature $T$ \[K\] follows from the specific entropy $S$ \[J/kg/K\] by $T = T_\mathrm{ref} \exp[(S - S_\mathrm{ref}) / c_p]$. Here $T_\mathrm{ref}$ = 3000 K and $S_\mathrm{ref}$ = 3000 J/kg/K are the reference values `const_T_ref` and `const_S_ref`, and $c_p$ = 1000 J/kg/K is the heat capacity `const_Cp` of the configuration below.

## 1. Write the configuration

The shell is 25 km thick, below the surface of an Earth-size planet. All of it starts at $T_i$ = 1100 K. The surface is held at $T_s$ = 1000 K. Convection is off, so heat moves by conduction only. The conductivity is $k$ = 4 W/m/K and $\rho c_p$ = 4 × 10⁶ J/m³/K, so the thermal diffusivity is $\kappa = k / (\rho c_p)$ = 10⁻⁶ m²/s.

```python
from pathlib import Path

config = """
[solver]
start_time = 0
end_time = 1e5
atol = 1e-8
rtol = 1e-8

[boundary_conditions]
outer_boundary_condition = 5        # prescribed surface temperature
outer_boundary_value = 1000.0       # K
inner_boundary_condition = 2        # prescribed flux at the bottom
inner_boundary_value = 0.0          # W/m^2: no heat crosses the bottom
emissivity = 1
equilibrium_temperature = 273
core_heat_capacity = 880
core_bc = "quasi_steady"

[mesh]
outer_radius = 6371000
inner_radius = 6346000
number_of_nodes = 101
mixing_length_profile = "nearest_boundary"
core_density = 10500
eos_method = 1
mass_coordinates = false

[energy]
conduction = true
convection = false
gravitational_separation = false
mixing = false
radionuclides = false
tidal = false
solver_method = "radau"
use_jax_jacobian = false

[initial_condition]
initial_condition = 1
surface_temperature = 1100
basal_temperature = 1100

[phase_liquid]
density = 4000
viscosity = 1e2
heat_capacity = 1000
melt_fraction = 1
thermal_conductivity = 4
thermal_expansivity = 3e-5

[phase_solid]
density = 4000
viscosity = 1e21
heat_capacity = 1000
melt_fraction = 0
thermal_conductivity = 4
thermal_expansivity = 3e-5

[phase_mixed]
latent_heat_of_fusion = 4e5
rheological_transition_melt_fraction = 0.4
rheological_transition_width = 0.15
solidus = "solidus.dat"
liquidus = "liquidus.dat"
phase = "mixed"
phase_transition_width = 0.01
grain_size = 1.0e-3
const_properties = true
const_rho = 4000
const_Cp = 1000
const_alpha = 3e-5
const_cond = 4
const_log10visc = 21
const_T_ref = 3000
const_S_ref = 3000
"""
Path('surface_temperature.toml').write_text(config)
```

The value of `outer_boundary_value` is a temperature in kelvin in this mode. In the prescribed-flux mode (`outer_boundary_condition = 4`) the same key is a flux in W/m².

## 2. Run the solver

The solver evolves the entropy $S$ in each cell. We give it the entropy of 1100 K as the start value in every cell, from the constant-properties values of the configuration. We pass `entropy_eos=None` because the constant-properties mode reads no EOS tables; it does not read the files `solidus.dat` and `liquidus.dat` named in `[phase_mixed]` either.

```python
import numpy as np

from aragog.parser import Parameters
from aragog.solver import EntropySolver

params = Parameters.from_file('surface_temperature.toml')
solver = EntropySolver(params, entropy_eos=None)
solver.initialize()
pm = params.phase_mixed
solver.set_initial_entropy(pm.const_S_ref + pm.const_Cp * np.log(1100.0 / pm.const_T_ref))
solver.solve()
out = solver.get_state()
print(f'surface node temperature: {float(np.ravel(out.T_basic)[-1]):.6f} K')
```

The surface node sits at the prescribed 1000 K. The solver sets the entropy of that node so that its temperature is $T_s$.

## 3. Compare the surface heat flux

For a half space that starts at $T_i$ and has its surface held at $T_s$, the surface heat flux at time $t$ is

$$F(t) = \frac{k\,(T_i - T_s)}{\sqrt{\pi \kappa t}}$$

(Turcotte and Schubert 2014, *Geodynamics*, 3rd ed., section 4.15, eq. 4.116, p. 189). Our shell is thin compared with the planet and deep compared with the cooled layer, so it behaves like a half space.

```python
from scipy.constants import Julian_year

k, kappa, t_i = 4.0, 1.0e-6, 1100.0
t_s = params.boundary_conditions.outer_boundary_value
t = params.solver.end_time * Julian_year  # s
f_exact = k * (t_i - t_s) / np.sqrt(np.pi * kappa * t)
f_top = float(np.ravel(out.heat_flux)[-1])
print(f'Aragog F_top     = {f_top:.6f} W/m^2')
print(f'half-space F(t)  = {f_exact:.6f} W/m^2')
print(f'relative difference = {f_top / f_exact - 1:+.2e}')
```

The two agree to 7.52 × 10⁻⁴. The estimate is the grid error (+1.248 × 10⁻³) minus the curvature effect (4.94 × 10⁻⁴), 7.54 × 10⁻⁴. The top cell is $h$ = 250 m thick, and the grid error of the flux is about 0.063 $(h/\delta)^2$ with $\delta = \sqrt{\kappa t}$; the shell, unlike a half space, is curved, and that lowers the flux by about $\sqrt{\pi \kappa t} / r_\mathrm{out}$, with $r_\mathrm{out}$ = 6371 km the outer radius.

## 4. Compare the temperature profile

The half-space temperature at depth $z$ below the surface is $T = T_s + (T_i - T_s)\,\mathrm{erf}\big(z / (2\sqrt{\kappa t})\big)$ (Turcotte and Schubert 2014, eq. 4.113, p. 188).

```python
from scipy.special import erf

depth = float(np.ravel(out.r_basic)[-1]) - np.ravel(out.r_stag)
t_exact = t_s + (t_i - t_s) * erf(depth / (2.0 * np.sqrt(kappa * t)))
t_model = np.ravel(out.T_stag)
print(f'largest difference: {np.max(np.abs(t_model - t_exact)):.3f} K')
print(f'cooled layer: T reaches 1090 K at {np.interp(1090.0, t_model[::-1], depth[::-1]) / 1e3:.2f} km depth')
```

The profile agrees with the error-function solution to 0.05 K. After 10⁵ years the temperature reaches 1090 K at about 4.1 km depth, so the cooled layer is thin compared with the 25 km shell.

## What to try next

- Change `outer_boundary_value` and check that $F$ scales with $T_i - T_s$.
- Use fewer nodes and watch the flux error grow as $(h/\delta)^2$.
- Read how the solver sets the surface node in [Surface boundary conditions](../Explanations/surface_bc.md), and the validation cases in [Surface boundary](../Validation/surface.md).
