# Properties at the solidus

In this tutorial you evaluate the heat capacity of the mantle just below and just above the solidus, with and without the blend of the material properties. You see why Aragog uses a blend width `matprop_smooth_width` above 0, and what the configuration does with a width of 0.

## What you will learn

- How to evaluate material properties at a given pressure and entropy with `EntropyPhaseEvaluator`.
- What the blend width does at the solidus.
- Which widths the configuration accepts.

## What you need

- Aragog installed (see [Installation](../How-to/installation.md)).
- The SPIDER P-S test tables, with `ARAGOG_TEST_EOS_DIR` set to their folder (see [Build tests](../How-to/build_tests.md)).

## 1. Compare the heat capacity on both sides of the solidus

The evaluator takes a pressure $P$ \[Pa\] and a specific entropy $S$ \[J/kg/K\] for each node and returns the blended properties. We evaluate it at 65.7 GPa, 10⁻⁶ J/kg/K below and above the solidus entropy $S_\mathrm{sol}$, for the widths 0 and 0.01.

```python
import os

import numpy as np

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator

eos = EntropyEOS(os.environ['ARAGOG_TEST_EOS_DIR'])
P = np.array([65.7e9])  # Pa, in the lower mantle
S_sol = float(eos.solidus_entropy(P)[0])
S_liq = float(eos.liquidus_entropy(P)[0])
print(f'solidus {S_sol:.1f} J/kg/K, liquidus {S_liq:.1f} J/kg/K')


def heat_capacity(width, S):
    """Heat capacity [J/kg/K] at pressure P and entropy S for a blend width."""
    ev = EntropyPhaseEvaluator(eos, gravitational_acceleration=10.0, matprop_smooth_width=width)
    ev.set_pressure(np.full_like(S, P[0]))
    ev.set_entropy(S)
    ev.update()
    return np.ravel(ev.heat_capacity())


S = S_sol + np.array([-1e-6, 1e-6])
for width in (0.0, 0.01):
    below, above = heat_capacity(width, S)
    print(f'width {width}: c_p below {below:.1f}, above {above:.1f}, ratio {above / below:.3f}')
```

The output is

```text
solidus 1350.2 J/kg/K, liquidus 9497.9 J/kg/K
width 0.0: c_p below 1373.5, above 29704.8, ratio 21.627
width 0.01: c_p below 15539.2, above 15539.2, ratio 1.000
```

At width 0 the heat capacity jumps by a factor of 21.6 between two entropies 2 × 10⁻⁶ J/kg/K apart: below the solidus it is the value of the solid table, above it the two-phase value, which holds the latent heat. At width 0.01 both sides have the mean of the two values. The evaluator called directly keeps a default width of 0; a run built from a configuration uses the width of the configuration.

## 2. Plot the blend

```python
import matplotlib.pyplot as plt

dS = np.linspace(-300.0, 300.0, 601)
for width, style in ((0.0, ':'), (0.01, '-')):
    plt.plot(dS, heat_capacity(width, S_sol + dS), style, label=f'width {width}')
plt.xlabel('S - S_sol [J/kg/K]')
plt.ylabel('c_p [J/kg/K]')
plt.legend()
plt.savefig('phase_blend.png')
```

The width is a fraction of the melting interval $S_\mathrm{liq} - S_\mathrm{sol}$ = 8148 J/kg/K at this pressure, so a width of 0.01 spreads the change over about 81 J/kg/K. The dotted line (width 0) steps at $S = S_\mathrm{sol}$; the solid line (width 0.01) passes smoothly through the mean value.

## 3. Check which widths the configuration accepts

The configuration classes refuse a width that is not strictly between 0 and 1.

```python
from aragog.config.phases import MixedPhaseConfig

required = dict(
    latent_heat_of_fusion=4e5,
    rheological_transition_melt_fraction=0.4,
    rheological_transition_width=0.15,
    solidus='solidus.dat',
    liquidus='liquidus.dat',
    phase='mixed',
    phase_transition_width=0.01,
    grain_size=1e-3,
)
print(MixedPhaseConfig(**required).matprop_smooth_width)
try:
    MixedPhaseConfig(**required, matprop_smooth_width=0.0)
except ValueError as err:
    print(err)
```

The output is

```text
0.01
matprop_smooth_width must lie in (0, 1), got 0.0
```

A TOML file read by `Parameters.from_file` refuses the same values. [Phase transitions](../Explanations/phase_transitions.md#width-of-the-blend-at-the-solidus-and-the-liquidus) explains the blend, and [Validation: Phase transitions](../Validation/phase_transitions.md) shows the checks.
