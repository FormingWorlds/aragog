# The melting point of a core alloy

This tutorial computes the liquidus of an iron alloy with light elements at core pressures with `aragog.core.AlloyLiquidus`: the eutectic points of the 3 binaries that lower the melting point, the liquidus of a mixed alloy, its slope, and the flags that mark extrapolated pressures and compositions past the eutectic. [The core alloy liquidus](../Explanations/core_alloy_liquidus.md) explains the model and [its validation](../Validation/core_alloy_liquidus.md) the checks against the source papers. The code needs no EOS tables.

## 1. The eutectic points

Each Fe-X binary has a eutectic temperature and composition at every pressure. At 135 GPa, near the core-mantle boundary of the Earth:

```python
from aragog.core import AlloyLiquidus, IronMeltingCurve, eutectic

for el in ('S', 'C', 'O'):
    t_e, w_e = eutectic(el, 135e9)
    print(f'Fe-{el} eutectic at 135 GPa: {float(t_e):.0f} K, {100 * float(w_e):.2f} wt% {el}')
```

```text
Fe-S eutectic at 135 GPa: 2672 K, 9.35 wt% S
Fe-C eutectic at 135 GPa: 2997 K, 3.77 wt% C
Fe-O eutectic at 135 GPa: 3180 K, 12.45 wt% O
```

## 2. The liquidus of a mixed alloy

An alloy with 2 wt% S, 0.5 wt% C, 3 wt% O and 4 wt% Si melts below pure iron. Si lowers no melting point; its atoms count with the iron.

```python
alloy = AlloyLiquidus({'S': 0.02, 'C': 0.005, 'O': 0.03, 'Si': 0.04})
for p_gpa in (135.0, 330.0):
    t_fe = float(IronMeltingCurve.t_melt_pure(p_gpa * 1e9))
    t_liq = float(alloy.t_melt(p_gpa * 1e9))
    print(f'{p_gpa:.0f} GPa: pure Fe {t_fe:.0f} K, alloy {t_liq:.0f} K, depression {t_fe - t_liq:.0f} K')
```

```text
135 GPa: pure Fe 4180 K, alloy 3567 K, depression 613 K
330 GPa: pure Fe 6229 K, alloy 4542 K, depression 1687 K
```

## 3. Slope and flags

The slope of the liquidus decides whether a core freezes from the centre or from the top; the flags mark a pressure outside the data of a binary present in the alloy, and a composition past the eutectic, whose liquidus is held at the eutectic value.

```python
import numpy as np

p = np.array([135e9, 330e9, 500e9])
slope = np.asarray(alloy.gradient(p)) * 1e9
flags = alloy.status(p)
for q, g, past, ext in zip(p / 1e9, slope, flags['past_eutectic'], flags['extrapolated']):
    print(f'{q:.0f} GPa: dT/dP {g:.2f} K/GPa, past eutectic {bool(past)}, extrapolated {bool(ext)}')
rich = AlloyLiquidus({'S': 0.12})
print(f"12 wt% S at 135 GPa: {float(rich.t_melt(135e9)):.0f} K, "
      f"past eutectic {bool(rich.status(np.array([135e9]))['past_eutectic'][0])}")
```

```text
135 GPa: dT/dP 7.69 K/GPa, past eutectic False, extrapolated False
330 GPa: dT/dP 3.20 K/GPa, past eutectic False, extrapolated True
500 GPa: dT/dP 5.76 K/GPa, past eutectic False, extrapolated True
12 wt% S at 135 GPa: 2672 K, past eutectic True
```

At 330 GPa the alloy is flagged because its O is outside the data: the Fe-O eutectic composition is read up to 325 GPa. 330 GPa is also the end of the S and C data, where the slope steps because the eutectic composition stops changing: the value printed there is the side inside the data, and at 330.001 GPa the slope is 5.48 K/GPa.

## 4. Adding a binary

A new light element that lowers the melting point needs 4 parts in `aragog/core/alloy.py` and its data folder `aragog/core/eutectics/`:

1. A table `fe_<el>_composition.csv` of the eutectic liquid composition (mass fraction) against pressure, with its source in the header.
2. The eutectic temperature: a Simon fit in `_SIMON`, or a table read like `fe_o_temperature.csv`.
3. The Fe_k X_m unit of the binary in `_UNITS`, and the element in `DEPRESSING` with its atomic weight in `_MOLAR`.
4. The pressure range of its data in `_data()`; an element whose temperature is a table, like O, also needs its branch in `_eutectic_data`.

Then add the binary to the end-point test `tests/test_core_alloy_liquidus.py::test_each_binary_runs_through_pure_iron_and_its_eutectic` and, if the source paper draws liquidus curves, a test like `::test_the_fe_s_liquidus_curves_of_mori_are_the_ideal_form_on_fe3s_units`.
