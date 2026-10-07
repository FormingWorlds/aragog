# Reference tables for the core verification page

Both tables are output of the `leeds` core model of [thermal_history](https://github.com/sam-greenwood/thermal_history) (Greenwood et al. 2021; MIT licence) at commit `ea9aa99`, made by `tools/verification/thermal_history_reference.py`. The first line of each holds, as JSON, the inputs, the package versions and the column names. Each row is the state at the start of a 1 Myr step, from a CMB temperature of 4400 K.

- `thermal_history_evolution.csv`: the core alone under a fixed 10 TW CMB heat flow for 1.5 Gyr, with the budget terms thermal_history evaluates on each state. The energy and entropy terms are per unit change of the central temperature, as thermal_history stores them.
- `thermal_history_stable_layer.csv`: the core with the `leeds_thermal` stable layer under fixed CMB heat flows of 8 and 12 TW for 1 Gyr, with the layer base `r_s`.
