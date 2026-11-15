# A core on the PALEOS liquid iron isentrope

This tutorial builds the core energy budget of a 1 Earth-mass planet from a Zalmoxis core and the PALEOS liquid iron table, with `profile_source = "paleos"`. Run the blocks from the root of the aragog repository: they read the test data in `tests/validation/data/`. No mantle EOS tables are needed.

[A core on the PALEOS liquid iron isentrope](../Explanations/core_tabulated.md) explains the model, and [its validation](../Validation/core_tabulated.md) shows the checks.

## 1. Build the budget

The core arrays come from a Zalmoxis structure: radius, enclosed mass, pressure, density and gravity at each core node. The table is a cut of the PALEOS liquid iron table: pressures from 45 GPa to 3 TPa, temperatures from 1996 K to 20,000 K. In a solver run the CMB radius comes from the mantle mesh; here it comes from the arrays.

```python
from aragog.core import CoreModule, build_core_module_budget, load_core_arrays

data = 'tests/validation/data/'
core = load_core_arrays(data + 'zalmoxis_core_arrays_1me_3000K.npz')
params = {
    'profile_source': 'paleos',
    'core_liquid_table': data + 'paleos_iron_liquid_cut_3tpa.npz',
    'core_profile_file': data + 'zalmoxis_core_arrays_1me_3000K.npz',
}
budget = build_core_module_budget(
    params, r_cmb=core['r'][-1], p_cmb_fallback=core['P'][-1], m_core=core['m'][-1]
)
prof = budget.profiles
print(f'nodes {prof.n_nodes}, core mass {prof.m_core:.4e} kg')
print(f'isentropes from {float(prof.t_grid[0]):.0f} to {float(prof.t_grid[-1]):.0f} K')
```

```text
nodes 81, core mass 1.9409e+24 kg
isentropes from 1996 to 12839 K
```

The budget tabulates the isentropes for every CMB temperature whose isentrope stays inside the table. The lowest is the lowest table temperature. The highest is the one whose centre reaches the highest table temperature, 20,000 K.

## 2. The isentrope and the secular heat capacity

This step prints the centre temperature and the secular heat capacity at 2 CMB temperatures.

```python
for t_cmb in (4000.0, 5000.0):
    print(f'T_cmb {t_cmb:.0f} K: T_cen {float(prof.t_cen(t_cmb)):.0f} K, '
          f'C_s {float(budget.secular_capacity(t_cmb=t_cmb)):.4e} J/K')
```

```text
T_cmb 4000 K: T_cen 6111 K, C_s 1.9888e+27 J/K
T_cmb 5000 K: T_cen 7676 K, C_s 1.9928e+27 J/K
```

The secular heat capacity $C_s$ is the heat the core gives up per kelvin of cooling at the CMB. It uses $c_p = 840$ J kg$^{-1}$ K$^{-1}$ by default. Set `'core_cp_override': 'table'` in `params` to take the table value at the CMB instead.

## 3. The inner core and the heat of a cooling step

This step prints the onset of the inner core and the heat the core gives up from 5000 to 4900 K.

```python
print(f'onset {budget.t_onset:.1f} K, freeze-out {budget.t_freeze:.1f} K')
print(f'heat change 5000 -> 4900 K: {budget.heat_change(5000.0, -100.0):.4e} J')
```

```text
onset 4186.5 K, freeze-out 4089.1 K
heat change 5000 -> 4900 K: -1.9927e+29 J
```

The onset is the CMB temperature at which the isentrope meets the pure-iron melting curve at the centre. Below the freeze-out temperature the CMB itself is below the melting curve. The heat change is negative: the core loses heat.

## 4. The diagnostics of the core

`CoreModule` holds the core temperature and reports its state.

```python
module = CoreModule(budget, t_cmb=5000.0)
diag = module.diagnostics()
print(f"inner core {diag['inner_core_present']}, "
      f"outside calibration {diag['core_eos_outside_calibration']:.2f}")
```

```text
inner core False, outside calibration 1.00
```

At a CMB of 5000 K the centre is at 7676 K (step 2), so the whole core is colder than the calibration window of the PALEOS liquid, 8000 to 14,000 K. The run continues; the value flags the extrapolation.

## 5. A table of your own

Any liquid iron table on a full rectangle in pressure and temperature works. Build it from arrays with `LiquidIronTable`. Each field is a 2-D array with one row per pressure, in SI units, and the header must name the liquid branch.

```python
import numpy as np
from aragog.core import LiquidIronTable

with np.load(data + 'paleos_iron_liquid_cut_3tpa.npz') as d:
    fields = {f: d[f] for f in ('rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad')}
    table = LiquidIronTable(d['P_Pa'], d['T_K'], fields, header='# Kind: liquid')
print(f"rho at 200 GPa, 5000 K: {table.value('rho', 200e9, 5000.0):.1f} kg/m3")
```

```text
rho at 200 GPa, 5000 K: 11867.7 kg/m3
```

Save the arrays with `np.savez_compressed` under the names `P_Pa`, `T_K`, the 7 field names and `header`, and point `core_liquid_table` to the file. Core arrays of your own go in an `.npz` with the arrays `r` (m), `m` (kg), `P` (Pa), `rho` (kg m$^{-3}$) and `g` (m s$^{-2}$), from the centre to the CMB; point `core_profile_file` to it.
