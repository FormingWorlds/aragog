# Solid-State Convection

Aragog supports simulating late-stage planetary cooling through solid-state convection in the mantle. While a fully molten mantle is governed by a purely liquid rheology, the interior transitions to solid-state flow as the melt fraction drops below the rheological transition ($\phi_\text{rheo}$). Aragog manages this transition by smoothly blending the liquid viscosity with an Arrhenius diffusion-creep viscosity, and supports plastic yielding to model mobile-lid and stagnant-lid tectonic regimes.

## Arrhenius Diffusion Creep

In the solid regime ($\phi < \phi_\text{rheo}$), the baseline viscosity follows an Arrhenius law for diffusion creep:


$$
\eta_\text{diff}(T, P) = \eta_\text{solid} \exp\left( \frac{E_a + P V_a}{R T} - \frac{E_a}{R T_\text{ref}} \right)
$$


where:
- $\eta_\text{solid}$ is the reference viscosity at the reference temperature $T_\text{ref}$ ($1600\text{ K}$, hardcoded).
- $E_a$ is the activation energy (`activation_energy`).
- $V_a$ is the activation volume (`activation_volume`).

At the cold surface, this Arrhenius viscosity becomes extremely high (diverging exponentially as $T \to 0$), leading to the formation of a rigid, non-convecting stagnant lid.

## Plastic Yielding

To permit a mobile lid (or episodic overturns) when convective stresses exceed the strength of the lithosphere, Aragog implements a Byerlee frictional yield criterion. The yield stress $\tau_y$ is parameterized as:


$$
\tau_y(P) = \tau_c + \mu P
$$


where $\tau_c$ is the cohesion (`yield_stress_c`) and $\mu$ is the coefficient of friction (`yield_stress_mu`). A maximum yield-stress ceiling of $500\text{ MPa}$ is enforced at depth (this is a fixed constant).

When the convective driving stress exceeds $\tau_y$, the effective solid viscosity is truncated. The yielded effective viscosity $\eta_\text{eff}$ is computed as a harmonic mean to ensure smooth, differentiable transitions:


$$
\eta_\text{eff} = \left( \frac{1}{\eta_\text{diff}} + \frac{2 \dot{\epsilon}}{\tau_y} \right)^{-1}
$$


where $\dot{\epsilon}$ is the convective strain rate.

## Strain Rate Closure Modes

Because the strain rate $\dot{\epsilon}$ depends on the convective velocity, which in turn depends on the viscosity $\eta_\text{eff}$, this poses a non-linear circularity. Aragog breaks this circularity by computing a *proxy* strain rate based on the **unyielded** baseline viscosity. This single-pass explicit closure guarantees numerical stability (avoiding solver divergence and infinite loops).

You can configure the geometric scope of this strain rate proxy via the `stress_closure_mode` parameter in the `PROTEUS` TOML config:

### `local` Mode
In `local` mode, the strain rate is evaluated purely from the local mixing-length velocity:

$$
\dot{\epsilon}_\text{local}(r) = \frac{|v_\text{unyielded}(r)|}{l(r)}
$$


**Note:** In a cold lithosphere, the unyielded Arrhenius viscosity is enormous, so $v_\text{unyielded} \approx 0$. Thus, $\dot{\epsilon}_\text{local} \approx 0$ and yielding never triggers. `local` mode only yields the deep, already-convecting mantle.

### `global` Mode
To capture lithospheric yielding driven by the convective vigor of the *entire* mantle (as in a boundary-layer model), `global` mode computes a bulk strain rate over the lithosphere thickness $d_\text{lid}$:

$$
\dot{\epsilon}_\text{global} = \frac{v_\text{interior}}{d_\text{lid}}
$$


The interior velocity $v_\text{interior}$ and lid thickness $d_\text{lid}$ are extracted from the unyielded temperature and velocity profiles using a hard $1400\text{ K}$ threshold (a fixed constant determining the lid base) and a `max`-based velocity selection. This applies a uniform tectonic driving strain rate to the full mantle, allowing a rigid lid to successfully yield when the underlying mantle convects vigorously enough.

## Configuration

In `PROTEUS` `config.toml`, you can set these parameters under the `[phase_solid]` block:

```toml
[phase_solid]
activation_energy = 300e3      # [J/mol]
activation_volume = 5e-6       # [m^3/mol]
yield_stress_c = 50e6          # [Pa] Cohesion
yield_stress_mu = 0.6          # Friction coefficient
stress_closure_mode = "global" # "local" or "global"
```

These parameters are identically passed to both the NumPy and JAX CVODE solvers. This preserves float64 parity to within ULP between evaluations.
