# Surface boundary conditions

How does the entropy solver close the heat flow at the top of the mantle?

The setting `outer_boundary_condition` in `[boundary_conditions]` picks the mode. The entropy solver implements 3 modes: 1, 4 and 5. The parser also accepts mode 2, for which the NumPy path applies no surface condition (the surface face carries the flux of the gradient copied from the cell below) and the JAX path reads `outer_boundary_value` as a flux, so do not use it; mode 3 is refused. Modes 1 and 4 set the heat flux through the surface. Mode 5 sets the temperature at the surface, and the heat flux then follows from the temperature difference between the surface and the top cell. The other surface values (`outer_boundary_value`, `emissivity`, `equilibrium_temperature`, `param_utbl`) are inputs of these modes.

## The three modes

| `outer_boundary_condition` | What it sets | `outer_boundary_value` | Typical use |
|---|---|---|---|
| 1 | Grey-body flux $F = \varepsilon \sigma (T_\mathrm{top}^4 - T_\mathrm{eq}^4)$ from the temperature of the surface node | not used | standalone runs with a simple atmosphere |
| 4 | The surface flux itself | flux \[W/m²\] | PROTEUS coupling: the atmosphere module gives the flux |
| 5 | The surface temperature | temperature \[K\] | benchmarks with a fixed surface temperature, such as stagnant-lid scaling cases |

Here $\varepsilon$ is the emissivity, $\sigma$ the Stefan-Boltzmann constant, $T_\mathrm{top}$ the temperature of the surface node and $T_\mathrm{eq}$ the equilibrium temperature. The [UTBL parameterisation](../How-to/utbl-parameterisation.md) only acts in mode 1. PROTEUS uses mode 4 (or mode 1); it does not use mode 5.

## Prescribed surface temperature (mode 5)

The solver works with the specific entropy $S$ \[J/kg/K\], not with the temperature $T$ \[K\]. Mode 5 must therefore turn a prescribed temperature into an entropy and an entropy gradient at the surface node.

**Short answer.** At setup the solver finds the entropy $S_s$ at which the equation of state (EOS) gives the prescribed temperature $T_s$ at the surface pressure. At every call it gives the surface node this entropy, and it sets the entropy gradient at the node so that the temperature gradient is the difference between $T_s$ and the top cell temperature over half a cell. The conductive flux through the surface is then exactly the half-cell conduction, and the convective flux uses the same gradient. Gravitational separation and chemical mixing carry no flux through the surface.

### The surface entropy

The mantle grid has cell centres (the staggered nodes, where the solver keeps $S$) and cell faces (the basic nodes, where the fluxes are). The surface is the last basic node. Its pressure is $P_s$ \[Pa\]. The solver solves $T(P_s, S_s) = T_s$ for $S_s$:

- with EOS tables, by a root search over the entropy range of the tables on the phase evaluator of the surface node, so that the value includes the property smoothing near the solidus and liquidus (`matprop_smooth_width`). On the test tables a smoothing width of 0.2 moves the root by up to 72 J/kg/K from the unsmoothed table value (at 1130 K, scanned from 330 to 2999 K);
- in the constant-properties mode, from the closed form $S_s = S_\mathrm{ref} + c_p \ln(T_s / T_\mathrm{ref})$, with $T_\mathrm{ref}$ \[K\] and $S_\mathrm{ref}$ \[J/kg/K\] the reference temperature and entropy of that mode (`const_T_ref`, `const_S_ref`).

### The surface gradient

The mantle has $N$ cells, numbered from 0 at the bottom. The centre of the top cell, number $N-1$, has the temperature $T_{N-1}$ \[K\]. It lies half a cell below the surface, at the distance $\Delta r / 2$, with $\Delta r$ \[m\] the width of the top cell. The temperature gradient across this half cell is

$$\left.\frac{\partial T}{\partial r}\right|_s = \frac{T_s - T_{N-1}}{\Delta r / 2}.$$

The solver needs an entropy gradient, not a temperature gradient. The two are related by the thermodynamic identity $(\partial T / \partial S)_P = T / c_p$, where $c_p$ is the heat capacity \[J/kg/K\]:

$$\frac{\partial T}{\partial r} = \frac{T}{c_p}\frac{\partial S}{\partial r} + \left(\frac{\partial T}{\partial P}\right)_S \frac{\partial P}{\partial r}.$$

So the solver sets the entropy gradient at the surface node to

$$\left.\frac{\partial S}{\partial r}\right|_s = \frac{c_p}{T_s}\left\lbrack\left.\frac{\partial T}{\partial r}\right|_s - \left(\frac{\partial T}{\partial P}\right)_S \frac{\partial P}{\partial r}\right\rbrack.$$

The second term is the adiabatic gradient. $T$ and $c_p$ are taken at the surface node, whose temperature is $T_s$. The heat capacity here has the same floor of 100 J/kg/K as the conductive flux, so the identity holds term by term.

### The surface flux

The conductive flux is $F_\mathrm{cond} = -k\,\partial T/\partial r$, with $k$ the thermal conductivity \[W/m/K\]. With the gradient above it becomes

$$F_\mathrm{cond} = k\,\frac{T_{N-1} - T_s}{\Delta r / 2},$$

positive outward when the mantle is warmer than the surface. The convective flux of the mixing-length theory, $\rho T \kappa_h (-\partial S / \partial r)$ with $\rho$ the density \[kg/m³\] and $\kappa_h$ the eddy diffusivity \[m²/s\], uses the same entropy gradient. Mode 5 needs `mixing_length_profile = 'nearest_boundary'`: the mixing length is 0 at the surface, so the convective flux there is 0, as at a wall where the eddy velocity vanishes. With `'constant'` the mixing length is a fixed share of the mantle depth everywhere (`mixing_length_constant_fraction`, 0.25 by default), and the surface face would carry an eddy flux, driven by the steep half-cell gradient, at a wall where none can pass; a negative `eddy_diffusivity_thermal` does the same, because it sets $\kappa_h$ at every node, the surface node too. Mode 5 refuses both. The floor of the eddy diffusivity (`kappah_floor`) lets molten cells mix below the convective threshold; it does not act at a surface held at its temperature.

### Why the solver does not use an entropy difference

A simpler rule would take the entropy gradient as $(S_s - S_{N-1}) / (\Delta r / 2)$. Entropy is not linear in temperature, so this rule gives the wrong flux. With constant $c_p$ it replaces $T_{N-1} - T_s$ by $T_s \ln(T_{N-1}/T_s)$. For a top cell at 380.46 K below a surface at 273 K that is 90.6 K instead of 107.5 K, a flux 16 % too low.

### The half-cell distance

The solver takes $\Delta r / 2 = \Delta\xi / (2\, \mathrm{d}\xi/\mathrm{d}r)$, where $\Delta\xi$ is the spacing of the top cell in the mesh coordinate $\xi$ and $\mathrm{d}\xi/\mathrm{d}r$ its derivative at the surface. On a mesh that is uniform in radius (`mass_coordinates = false`) this equals the distance from the top cell centre to the surface. On a mass-coordinate mesh the two differ: on a 19-cell Earth mantle the half cell is 91,797 m and the geometric distance 91,376 m, 0.46 % apart. The solver uses the first value on both meshes.

### Limits

- `outer_boundary_value` must be a positive finite number in kelvin. A string, a TOML `true`, NaN, an infinity, zero or a negative value stops the setup with a `ValueError`.
- A temperature outside the range the phase evaluator reaches at the surface pressure stops the setup with a `ValueError` that gives the range and the smoothing width; the solver does not clamp it to the table edge. With smoothing this range can be narrower than the tables: at a width of 0.5 the test tables start at 363 K instead of 320 K.
- A smoothing width of 0.1 or more makes the smoothed temperature jump where the melt fraction is 0.5: on the test tables by 0.02 K at a width of 0.1, 0.5 K at 0.15 and 2.6 K at 0.2 (from 1633.8 to 1636.4 K). No entropy gives a temperature inside the jump, so such a surface temperature stops the setup with a `ValueError`.
- These settings stop the setup with a `ValueError`:
    - `param_utbl = true`: the UTBL correction belongs to the grey-body flux.
    - `core_bc = 'gradient'`: the gradient core keeps its own surface gradient.
    - `energy.conduction = false`: without conduction the prescribed temperature sets no flux.
    - a mixing length that is not 0 at the surface node, as with `mixing_length_profile = 'constant'`, and a negative `eddy_diffusivity_thermal`: both put an eddy flux on the surface face.
- On the JAX path, `BoundaryParams` refuses `param_utbl = true` with type 5, and the JAX CVODE factory refuses conduction off, a mixing length that is not 0 at the surface node and a negative eddy diffusivity. `EntropySolver` turns an error of the factory into a warning and falls back to the NumPy right-hand side, but its own setup refuses these settings first. A direct caller of the JAX right-hand side or of `solve_entropy` is not checked for conduction and the eddy flux.
- A caller can change `outer_boundary_value` between calls; the next call of the NumPy right-hand side, and the next `solve`, invert the surface entropy again.
- Every call of the NumPy right-hand side checks that the surface node is at $T_s$ to a relative 10⁻⁶ and raises a `RuntimeError` if it is not.
- The JAX right-hand side takes the surface entropy and the half cell as `S_surf` and `surf_dr_half` of `BoundaryParams` with `outer_bc_type = 5`, and applies the same gradient, with the temperature of the surface node at `S_surf` in place of $T_s$. `S_surf` alone sets the surface temperature: `outer_bc_value` is only checked to be a positive finite number, so a new temperature needs a new `S_surf`, and a stale `outer_bc_value` raises no error on this path. `EntropySolver.surface_temperature_bc` gives the surface entropy, the surface temperature (`outer_bc_value`) and the half cell; build `BoundaryParams` again after a change of `outer_boundary_value`. `BoundaryParams` refuses type 5 without a finite `S_surf` and a positive finite `outer_bc_value` and `surf_dr_half`.

**Where this is checked.** [Surface boundary](../Validation/surface.md) compares mode 5 with a steady heated shell, a cooling half space, and a grid-convergence study with convection, notes the comparison on the stored states of the stagnant-lid benchmarks, and the [tutorial](../Tutorials/surface_temperature.md) repeats the half-space case step by step.
