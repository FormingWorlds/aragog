# Adams-Williamson pressure EOS

Aragog supports two pressure equations of state for the static mesh background:

| `mesh.eos_method` | Class | Use case |
|-------------------|-------|----------|
| `1` | `AdamsWilliamsonEOS` (default) | SPIDER-parity exponential profile derived from surface density and bulk modulus |
| `2` | `UserDefinedEOS` | External four-column file (`r`, `P`, `rho`, `g`) supplied by Zalmoxis or another structure model |

Both are implemented in [`aragog.mesh.pressure_eos`](../Reference/api/aragog.mesh.md). This page explains the analytic form of the Adams-Williamson EOS and the Newton solve used to invert it for the mass-coordinate mesh.

## Analytic form

The Adams-Williamson EOS assumes hydrostatic balance with a constant adiabatic bulk modulus $K_S$ and gravity $g$. Combined, these give the closed-form exponential density profile

$$
\rho(z) = \rho_s \exp(\beta z), \qquad \beta = \frac{\rho_s g}{K_S}
$$

with $z = R_\text{surf} - r$ the depth and $\rho_s$ the surface density. The pressure follows by integration of $dP/dr = -g \rho$:

$$
P(r) = \frac{\rho_s g}{\beta}\left(e^{\beta z} - 1\right) + P_\text{surf}
$$

which is invertible analytically:

$$
r(P) = R_\text{surf} - \frac{1}{\beta} \ln\!\left(1 + \frac{(P - P_\text{surf}) \beta}{\rho_s g}\right)
$$

The solver uses the analytic inverse directly; no Newton iteration is required for $r(P)$.

This matches SPIDER's [`eos_adamswilliamson.c`](https://github.com/FormingWorlds/SPIDER) line by line. The previous Aragog implementation used a rational (hyperbolic) form $\rho = \rho_s K_S / (K_S - \rho_s g z)$ derived from the same $K_S$ but differing from SPIDER by up to 6 % at CMB depth, causing 3 % $R_\text{int}$ mismatches on the same target planet mass. Switching to the exponential form closed that gap.

## Beta override

`mesh.adams_williamson_beta` lets the user pass $\beta$ directly. When the override is positive, it takes precedence over the $K_S$-derived value; when zero (default), the EOS computes $\beta = \rho_s g / K_S$. The override is the normal path for SPIDER-parity runs because SPIDER reads $\beta$ from its own input file.

## Mass coordinates

When `mesh.mass_coordinates = true` the basic-node radii are spaced uniformly in mass coordinate $\xi$ rather than radius $r$. The mass coordinate is

$$
\xi(r) = \frac{m(r) - m(R_\text{cmb})}{m(R_\text{surf}) - m(R_\text{cmb})}
$$

with $m(r)$ the cumulative mass enclosed by radius $r$. For the Adams-Williamson EOS the antiderivative is closed-form:

$$
M(r) = 4\pi \rho(r) \left( -\frac{2}{\beta^3} - \frac{r^2}{\beta} - \frac{2r}{\beta^2} \right) + C
$$

(matching SPIDER's `eos_adamswilliamson.c` line 191). The constant $C$ cancels in any difference $M(r_2) - M(r_1)$, so anchoring at $r_\text{cmb}$ is sufficient.

For uniformly spaced $\xi$, the corresponding $r$ values are found by inverting $M(r)$ via `scipy.optimize.brentq`, bracketed between the surface and CMB. brentq is preferred over Newton because $M(r)$ is strictly monotonic and the bracketed root is guaranteed.

## User-defined EOS path

When `eos_method = 2`, the EOS reads a four-column whitespace-separated file with columns `radius`, `pressure`, `density`, `gravity`. Aragog interpolates each onto the basic-node radii using `scipy.interpolate.PchipInterpolator` (shape-preserving cubic). The cumulative mass is precomputed via `scipy.integrate.cumulative_trapezoid` and then PCHIP-interpolated as well, so `get_mass_within_radii` is $O(1)$ per call.

Validation: at construction the EOS asserts that the user file's radius range covers at least 75 % of the mesh's $[R_\text{cmb}, R_\text{surf}]$ span; otherwise it raises a `ValueError`. This guards against an undersized external mesh silently zero-padding the deep interior.

## When the closed form is wrong

The Adams-Williamson assumption (constant $K_S$ at the relevant depth) breaks down for super-Earths and sub-Neptunes where the bulk modulus varies by an order of magnitude across the mantle. For those planets the user-defined path is the correct choice; Zalmoxis exports a four-column file that captures the full EOS variation and feeds it directly to Aragog. See the [PROTEUS framework](../proteus-framework.md) entry for the coupled use.
