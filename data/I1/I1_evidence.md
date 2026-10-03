# Stagnant Lid Driving Stress Closures: Verification and Evidence Report

Author: Tim Lichtenberg
Date: 2026-10-03
Worktree: `/Users/timlichtenberg/work/stream-ssc-step1/wt-repairs`, branch `tl/ssc-step1-repairs`
Associated artifacts:
- `data/I1/tau_d_comparison_table.csv`
- `data/I1/tau_d_sweep.csv`
- `data/I1/tau_d_vs_eta_bulk.png`

---

## 1. Five-Option Stress Comparison

### 1.1 Stress Formulations

1. **Current Code**:
   $$\tau_d = \frac{\eta_i v_i}{\delta_\mathrm{rh}}$$
   The convective velocity, $v_i$, is calculated using bulk mixture viscosity, $\nu = \eta_\mathrm{bulk} / \rho$, in mixing length theory. The dynamic viscosity, $\eta_i$, is the solid Arrhenius viscosity.

2. **Option (a) Consistent Viscosity**:
   $$\tau_d = \frac{\eta_\mathrm{bulk} v_i}{\delta_\mathrm{rh}}$$
   Calculates shear stress with the same viscosity, $\eta_\mathrm{bulk}$, that determines the convective velocity, $v_i$.

3. **Option (b) Solid Velocity Scale**:
   $$\tau_d = \frac{\eta_i v_{i,\mathrm{solid}}}{\delta_\mathrm{rh}}$$
   Calculates convective velocity from the solid diffusion creep viscosity, $\eta_i$, instead of the melt-weakened bulk mixture. In the viscous branch of mixing length theory, $v_{i,\mathrm{solid}} = v_i (\eta_\mathrm{bulk} / \eta_i)$, which gives the identical stress as Option (a).

4. **Option (b$^\prime$) Published Stagnant-Lid Convective Stress Scale**:
   Foley & Bercovici (2014, GJI 199, p. 588, Eq. 28 after Eq. 26) define the interior shear stress:
   $$\tau_{xz} = \frac{2 \mu_\mathrm{eff} v_m}{d}$$
   with $\mu_\mathrm{eff} = \mu_i$ (undamaged solid viscosity). The factor of 2 arises because the horizontal velocity goes from $v_m$ at the boundary to zero at mid-depth ($z = d/2$), as explained on pages 584 (lines 410-412) and 588 (lines 871-872).
   The interior convective velocity, $v_m$, follows the boundary layer scaling of Dumoulin et al. (1999) and Foley & Bercovici (2014, p. 587, Eq. 26):
   $$v_m = \frac{\kappa}{d} C_4 \left( \frac{\mathrm{Ra}_\mathrm{eff} a_\mathrm{rh}}{\theta} \right)^{2/3}$$
   where $C_4 = 0.125$, $a_\mathrm{rh} = 1.3$, $d$ is layer depth, and $\theta = E \Delta T / (R T_i^2)$ is the Frank-Kamenetskii parameter.
   The 1/2-exponent variant of Solomatov & Moresi (2000, JGR 105, p. 21804, Eq. 28) for internally heated convection is:
   $$v_m = \frac{\kappa}{d} \cdot 0.38 \left( \frac{\mathrm{Ra}_\mathrm{eff}}{\theta} \right)^{1/2}, \quad \tau_{xz} = \frac{2 \mu_i v_m}{d}$$

5. **Option (c) Rheological Sublayer Buoyancy Stress**:
   $$\tau_\mathrm{buoy} = \rho g \alpha \Delta T_\mathrm{rh} \delta_\mathrm{rh}$$
   where $\rho$ is density, $g$ is gravity, $\alpha$ is thermal expansivity, $\Delta T_\mathrm{rh} = a_\mathrm{rh} R T_i^2 / E$ is the rheological temperature scale, and $\delta_\mathrm{rh}$ is sublayer thickness.

---

### 1.2 Comparison Table: Solid Earth and Cold-Top-Lid Fixture

Data from `data/I1/tau_d_comparison_table.csv`:

| Case | Option | $\rho$ [kg/m$^3$] | $g$ [m/s$^2$] | $\alpha$ [K$^{-1}$] | $\Delta T_\mathrm{rh}$ [K] | $\delta_\mathrm{rh}$ [m] | $d$ [m] | $\kappa$ [m$^2$/s] | $\theta$ | $\mathrm{Ra}_\mathrm{eff}$ | $a_\mathrm{rh}$ | $\eta_\mathrm{bulk}$ [Pa s] | $\eta_i$ [Pa s] | $v_\mathrm{conv}$ [m/s] | $\tau_d$ [MPa] | $\tau_y$ [MPa] | Yields? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Solid Earth** | Current Code | 3300.0 | 9.81 | 3.00e-5 | 68.0 | 6800.0 | 2.89e6 | 1.00e-6 | 18.32 | 3.05e8 | N/A | 1.00e20 | 1.00e20 | 7.92e-11 | 1.1647 | 50.0 | NO |
| **Solid Earth** | Option (a) | 3300.0 | 9.81 | 3.00e-5 | 68.0 | 6800.0 | 2.89e6 | 1.00e-6 | 18.32 | 3.05e8 | N/A | 1.00e20 | 1.00e20 | 7.92e-11 | 1.1647 | 50.0 | NO |
| **Solid Earth** | Option (b) | 3300.0 | 9.81 | 3.00e-5 | 68.0 | 6800.0 | 2.89e6 | 1.00e-6 | 18.32 | 3.05e8 | N/A | 1.00e20 | 1.00e20 | 7.92e-11 | 1.1647 | 50.0 | NO |
| **Solid Earth** | (b$^\prime$) FB2014 ($2/3$, mantle $\Delta T=1300\mathrm{\ K}$) | 3300.0 | 9.81 | 3.00e-5 | 68.0 | 6800.0 | 2.89e6 | 1.00e-6 | 18.32 | 3.05e8 | 1.30 | 1.00e20 | 1.00e20 | 3.36e-9 | 0.2323 | 50.0 | NO |
| **Solid Earth** | (b$^\prime$) FB2014 ($2/3$, sublayer $\Delta T=130\mathrm{\ K}$) | 3300.0 | 9.81 | 3.00e-5 | 68.0 | 6800.0 | 2.89e6 | 1.00e-6 | 18.32 | 3.05e7 | 1.30 | 1.00e20 | 1.00e20 | 7.23e-10 | 0.0500 | 50.0 | NO |
| **Solid Earth** | (b$^\prime$) SM2000 ($1/2$, mantle $\Delta T=1300\mathrm{\ K}$) | 3300.0 | 9.81 | 3.00e-5 | 68.0 | 6800.0 | 2.89e6 | 1.00e-6 | 18.32 | 3.05e8 | N/A | 1.00e20 | 1.00e20 | 5.36e-10 | 0.0371 | 50.0 | NO |
| **Solid Earth** | (b$^\prime$) SM2000 ($1/2$, sublayer $\Delta T=130\mathrm{\ K}$) | 3300.0 | 9.81 | 3.00e-5 | 68.0 | 6800.0 | 2.89e6 | 1.00e-6 | 18.32 | 3.05e7 | N/A | 1.00e20 | 1.00e20 | 1.70e-10 | 0.0117 | 50.0 | NO |
| **Solid Earth** | Option (c) Buoyancy | 3300.0 | 9.81 | 3.00e-5 | 68.0 | 6800.0 | 2.89e6 | 1.00e-6 | 18.32 | 3.05e8 | N/A | 1.00e20 | 1.00e20 | N/A | 0.4491 | 50.0 | NO |
| **Cold Top Lid** | Current Code | 3157.4 | 9.81 | 4.18e-5 | 102.3 | 9878.4 | 1.00e6 | 1.00e-6 | 14.44 | 4.26e7 | N/A | 3.11e7 | 4.86e19 | 5396.29 | 2.6546e13 | 0.1 | YES (unphysical) |
| **Cold Top Lid** | Option (a) | 3157.4 | 9.81 | 4.18e-5 | 102.3 | 9878.4 | 1.00e6 | 1.00e-6 | 14.44 | 4.26e7 | N/A | 3.11e7 | 4.86e19 | 5396.29 | 16.9781 | 0.1 | YES |
| **Cold Top Lid** | Option (b) | 3157.4 | 9.81 | 4.18e-5 | 102.3 | 9878.4 | 1.00e6 | 1.00e-6 | 14.44 | 4.26e7 | N/A | 3.11e7 | 4.86e19 | 3.45e-9 | 16.9766 | 0.1 | YES |
| **Cold Top Lid** | (b$^\prime$) FB2014 (solid $\mu_i = \eta_i$) | 3157.4 | 9.81 | 4.18e-5 | 102.3 | 9878.4 | 1.00e6 | 1.00e-6 | 14.44 | 4.26e7 | 1.30 | 3.11e7 | 4.86e19 | 3.06e-9 | 0.2975 | 0.1 | YES (at $\tau_y=0.1\mathrm{\ MPa}$; NO at $\tau_y \ge 0.5\mathrm{\ MPa}$) |
| **Cold Top Lid** | (b$^\prime$) FB2014 (mushy $\mu_i = \eta_\mathrm{bulk}$) | 3157.4 | 9.81 | 4.18e-5 | 102.3 | 9878.4 | 1.00e6 | 1.00e-6 | 14.44 | 6.66e19 | 1.30 | 3.11e7 | 4.86e19 | 4.12e-1 | 2.56e-5 | 0.1 | NO ($\mathrm{Ra} \gg 5 \times 10^7$) |
| **Cold Top Lid** | (b$^\prime$) SM2000 (solid $\mu_i = \eta_i$) | 3157.4 | 9.81 | 4.18e-5 | 102.3 | 9878.4 | 1.00e6 | 1.00e-6 | 14.44 | 4.26e7 | N/A | 3.11e7 | 4.86e19 | 6.52e-10 | 0.0634 | 0.1 | NO |
| **Cold Top Lid** | Option (c) Buoyancy | 3157.4 | 9.81 | 4.18e-5 | 102.3 | 9878.4 | 1.00e6 | 1.00e-6 | 14.44 | 4.26e7 | N/A | 3.11e7 | 4.86e19 | N/A | 1.3088 | 0.1 | YES (at $\tau_y=0.1\mathrm{\ MPa}$; NO at $\tau_y \ge 1.5\mathrm{\ MPa}$) |

---

### 1.3 Solid-Limit Ratios and Length Scales

1. **Solid Earth Agreement**:
   - Option (c) gives $\tau_\mathrm{buoy} = 0.4491\mathrm{\ MPa}$.
   - Option (b) gives $\tau_d = 1.1647\mathrm{\ MPa}$. The ratio (b) / (c) is $2.59$, satisfying order-1 agreement.
   - Option (b$^\prime$) FB2014 with sublayer $\Delta T = 130\mathrm{\ K}$ gives $0.0500\mathrm{\ MPa}$. The ratio (c) / (b$^\prime$) is $8.98 \approx 9\times$.
   - Option (b$^\prime$) FB2014 with whole-mantle $\Delta T = 1300\mathrm{\ K}$ gives $0.2323\mathrm{\ MPa}$. The ratio (c) / (b$^\prime$) is $1.93 \approx 2\times$.
   - Option (b$^\prime$) SM2000 variant with sublayer $\Delta T = 130\mathrm{\ K}$ gives $0.0117\mathrm{\ MPa}$. The ratio (c) / (b$^\prime$) is $38.3 \approx 38\times$.

2. **Cold-Top-Lid Fixture Agreement**:
   - Option (c) gives $1.3088\mathrm{\ MPa}$.
   - Option (b) gives $16.9766\mathrm{\ MPa}$. The ratio (b) / (c) is $12.97 \approx 13\times$ ($> 10$).
   - Option (b$^\prime$) FB2014 (solid $\mu_i$) gives $0.2975\mathrm{\ MPa}$. The ratio (c) / (b$^\prime$) is $4.40$.

3. **Length Scale Comparison**:
   - Options (b) and (c) use the rheological boundary layer thickness, $\delta_\mathrm{rh}$ ($6.8\mathrm{\ km}$ for Earth, $9.88\mathrm{\ km}$ for fixture).
   - Option (b$^\prime$) uses the whole-layer depth, $d$ ($2890\mathrm{\ km}$ for Earth, $1000\mathrm{\ km}$ for fixture).
   - The choice of length scale explains a factor of $d / \delta_\mathrm{rh} = 2890 / 6.8 \approx 425 \sim 430$ for Earth.

---

### 1.4 Mushy Sub-Lid Layer Viscosity and Tested Rayleigh Number Range

When the layer beneath the lid contains melt ($\phi > 0$), the bulk viscosity is $\eta_\mathrm{bulk}$.
In a single-material model, the convective layer viscosity at $T_i$ is $\mu_i = \eta_\mathrm{bulk}$:
- In the cold-top-lid fixture, $\eta_\mathrm{bulk} = 3.108 \times 10^7\mathrm{\ Pa\ s}$.
- With $\mu_i = \eta_\mathrm{bulk}$, the effective Rayleigh number is:
  $$\mathrm{Ra}_\mathrm{eff} = \frac{\rho g \alpha \Delta T d^3}{\kappa \eta_\mathrm{bulk}} = 6.66 \times 10^{19}$$
  The interior velocity from FB2014 Eq. 26 is $v_m = 0.412\mathrm{\ m/s}$, and the shear stress is $\tau_{xz} = 2.56 \times 10^{-5}\mathrm{\ MPa}$ ($25.6\mathrm{\ Pa}$).
- In a magma ocean where $\eta_\mathrm{bulk} = 10^2\mathrm{\ Pa\ s}$, $\mathrm{Ra}_\mathrm{eff} = 2.07 \times 10^{25}$.
- **Tested Range in FB2014**: Foley & Bercovici (2014, Section 4.2, p. 589; Table 2, p. 593; Table 3, p. 594; Figures 6, 8, 9, p. 588-589) tested numerical models in the stagnant lid regime over:
  $$\mathrm{Ra}_0 \in [10^5, 5 \times 10^7]$$
  Mushy sub-lid conditions ($\mathrm{Ra}_\mathrm{eff} \sim 10^{19} - 10^{25}$) are **12 to 17 orders of magnitude** above the tested range of FB2014. The solid-state creep scaling laws break down in liquid-rich mushy layers where convection is turbulent.

---

### 1.5 Convective Velocity Exponent: 2/3 vs 1/2

Foley & Bercovici (2014, p. 587, Section 4.1) compare the velocity exponents:
> "Solomatov & Moresi (2000) find that $v \sim (\mathrm{Ra}_\mathrm{eff}/\theta)^{1/2}$ for their internally heated results. However, they also state that (26) fits the bottom heated experiments of Dumoulin et al. (1999), and that the 1/2 power-law scaling may be a transitional regime found at low Rayleigh numbers. We find that (26) is the correct scaling for our results."

**Application to a cooling post-magma-ocean mantle (estimates)**:
- In models of magma ocean solidification, basal crystallization transfers primordial core heat into the base of the newly formed solid mantle (e.g. Solomatov 2000, 2007; estimate).
- The estimated thermal Rayleigh number of the newly solidified mantle ($\mathrm{Ra} \sim 10^7 - 10^9$, order-of-magnitude estimate from mantle thickness $d \approx 2890\mathrm{\ km}$, $\Delta T \approx 1000 - 1500\mathrm{\ K}$, and initial solid viscosity $\eta \approx 10^{20} - 10^{21}\mathrm{\ Pa\ s}$) is well above the transitional low-$\mathrm{Ra}$ regime ($\mathrm{Ra} \lesssim 10^5$, Solomatov & Moresi 2000).
- Therefore, the $2/3$ exponent ($v_m \propto \mathrm{Ra}^{2/3}$, Dumoulin et al. 1999; FB2014 Eq. 26) applies to a cooling post-magma-ocean mantle.

---

### 1.6 Parameter Sweep: $\eta_\mathrm{bulk}$ from $10^2$ to $10^{21}\mathrm{\ Pa\ s}$

Data from `data/I1/tau_d_sweep.csv`, plotted in `data/I1/tau_d_vs_eta_bulk.png`:
- **Current Code**: Diverges from $16.98\mathrm{\ MPa}$ at $\eta_\mathrm{bulk} = 10^{20}\mathrm{\ Pa\ s}$ to $2.65 \times 10^{19}\mathrm{\ Pa}$ at $\eta_\mathrm{bulk} = 10^2\mathrm{\ Pa\ s}$, because $\eta_i$ remains solid ($4.86 \times 10^{19}\mathrm{\ Pa\ s}$) while mixing length velocity approaches free fall ($5396\mathrm{\ m/s}$).
- **Option (c) Buoyancy**: Strictly constant at $\tau_\mathrm{buoy} = 1.3088\mathrm{\ MPa}$ across the entire range from solid ($10^{21}\mathrm{\ Pa\ s}$) to melt ($10^2\mathrm{\ Pa\ s}$). It is continuous, everywhere finite, and bounded.
- **Option (b$^\prime$) FB2014 (solid $\mu_i = \eta_i$)**: Constant at $0.2975\mathrm{\ MPa}$, bounded and well below rock yield strength.

---

## 2. Literature Sources Read in Original PDFs

### 2.1 Moresi & Solomatov (1998, GJI 133, p. 677, Eq. 21-22)
> "In the steady state, we observed that the velocity of the upper boundary layer is controlled by the interior viscosity, so the appropriate stress scale is:
> $$\tau \sim \frac{\eta_i u_0}{d} \quad (21)$$
> in which the upper boundary layer velocity, $u_0$, is (from the energy equation)
> $$u_0 \sim \frac{\kappa d}{\delta_0^2} \quad (22)$$
> where $\delta_0$ is the upper boundary layer thickness which we determine from the Nusselt number ($\delta_0 \sim d \mathrm{Nu}^{-1} \sim d \mathrm{Ra}_i^{-1/3}$)."

- **Length Scale**: Moresi & Solomatov (1998) use the layer depth, $d$, in the denominator of the stress scaling ($\tau \sim \eta_i u_0 / d$), not the boundary layer thickness $\delta_0$.

### 2.2 Solomatov (1995, Phys. Fluids 7, p. 268, Eq. 24, 28)
> "Note, that instability occurs not in the entire boundary layer but only in a thin sublayer near the bottom, provided $e^p > e^{8} \approx 3 \times 10^3$. The thickness of the unstable sublayer is equal to:
> $$z_\mathrm{sub} = \frac{8 \delta_0}{p} \quad (28)$$
> where $\Delta T_\mathrm{sub} \sim p^{-1} \Delta T$ (Eq. 24)."

Here, $p = \theta / \Delta T$, giving $\Delta T_\mathrm{sub} = \Delta T / \theta = R T_i^2 / E$, defining the rheological temperature scale of the unstable sublayer.

### 2.3 Solomatov & Moresi (2000, JGR 105, p. 21800, Sec. 5.1)
The temperature drop across the rheological sublayer is:
$$\Delta T_\mathrm{rh} = T_i - T_L = a_\mathrm{rh} \frac{R T_i^2}{E}$$
with $a_\mathrm{rh} \approx 2.2$ for Newtonian convection with strongly temperature-dependent viscosity.

### 2.4 Foley & Bercovici (2014, GJI 199, p. 588, Eq. 28)
> "We also assume that $\tau_{xz}$ is the dominant stress component in the convecting interior, and that $\tau_{xz} = (2 \mu_\mathrm{eff} v_m)/d$."

The sub-lid non-dimensional shear stress in Table 2 (p. 593) ranges from $\tau'_{xz} = 165$ to $1253$. For Earth mantle scales ($\mu_m = 10^{20}\mathrm{\ Pa\ s}$, $d = 2890\mathrm{\ km}$, $\kappa = 10^{-6}\mathrm{\ m^2/s}$, stress unit $\mu_m \kappa / d^2 = 11.97\mathrm{\ Pa}$), dimensional convective stresses are $0.002$ to $0.015\mathrm{\ MPa}$ for laminar models, and reach $0.02$ to $2.0\mathrm{\ MPa}$ at higher $\mathrm{Ra}_0$ and damage in Figures 8(c) and 9(c) (p. 588-589).

### 2.5 Laboratory Rock Yield Stresses
- Byerlee (1978, Pure Appl. Geophys. 116, pp. 615, 624-625; Eq. 2 on p. 625):
  $$\tau = 50\mathrm{\ MPa} + 0.6 \bar{\sigma}_n \quad (200 < \bar{\sigma}_n \le 1700\mathrm{\ MPa})$$
  At lithospheric depths $z = 10 - 20\mathrm{\ km}$, lithostatic confining pressure is $P = \rho g z \approx 300 - 600\mathrm{\ MPa}$ (with $\rho \approx 3000\mathrm{\ kg/m^3}$ and $g = 9.81\mathrm{\ m/s^2}$). Byerlee's law gives yield stresses of $\tau_y \approx 230 - 410\mathrm{\ MPa}$ (commonly quoted as $200 - 500\mathrm{\ MPa}$; Byerlee 1978, p. 625; Kohlstedt et al. 1995).
- Foley et al. (2012, EPSL 331-332, pp. 281-282, Sec. 1.1):
  > "there is a large mismatch between experimentally determined lithospheric strength and convective stress, meaning that unrealistically low yield stress values are necessary to produce convection with a mobile plate-like surface (e.g. Moresi and Solomatov, 1998; Tackley, 2000b)."

### 2.6 Model Closure Statement
No published paper writes the lid driving stress as $\tau_d = \rho g \alpha \Delta T_\mathrm{rh} \delta_\mathrm{rh}$ directly.
Option (c) is an Aragog model closure balancing the negative thermal buoyancy force of the rheological sublayer. The documentation in `docs/Explanations/solid_state_convection.md` states this clearly.

---

## 3. Yield Stress Mobilization and Laboratory Scales

1. **Convective Stress vs Laboratory Strength**:
   - Convective driving stresses are $\tau_\mathrm{buoy} \approx 0.45\mathrm{\ MPa}$ (Earth) and $1.31\mathrm{\ MPa}$ (fixture).
   - Laboratory rock yield stresses from Byerlee (1978, p. 625, Eq. 2) are $200 - 500\mathrm{\ MPa}$ at lithospheric confining pressures ($200 - 1700\mathrm{\ MPa}$).
   - The ratio is:
     $$\frac{\tau_\mathrm{buoy}}{\tau_{y,\mathrm{lab}}} \sim \frac{1\mathrm{\ MPa}}{300\mathrm{\ MPa}} \approx 0.003 \ll 1$$

2. **Regime Switch Assessment**:
   - Under realistic laboratory rock yield stresses ($\tau_y \ge 100\mathrm{\ MPa}$; Byerlee 1978, p. 625; Kohlstedt et al. 1995), convective driving stress $\tau_d$ in a stagnant lid is orders of magnitude below $\tau_y$.
   - The Step 1 plastic yielding regime switch can **never fire** for laboratory rock yield stresses.
   - The regime switch fires only when an artificially reduced pseudoplastic yield stress is specified ($\tau_y \lesssim 1.5\mathrm{\ MPa}$ for the fixture, or $\tau_y \lesssim 1.0\mathrm{\ MPa}$ for Earth), consistent with the qualitative finding of Foley et al. (2012, pp. 281-282) that pseudoplastic models require unrealistically low yield stresses.

---

## 4. Cold-Top-Lid Fixture Flux Decomposition at Top 3 Faces

Simulations were performed on `tests/configs/cold_top_lid.toml` at $t = 0.10\mathrm{\ yr}$.
The top 3 basic face radii are:
- Face [-3]: $r = 6258.52\mathrm{\ km}$ (sub-lid boundary)
- Face [-2]: $r = 6314.55\mathrm{\ km}$ (interior lid face)
- Face [-1]: $r = 6371.00\mathrm{\ km}$ (outer surface boundary face)

### 4.1 Numerical Decomposition

1. **Yielding OFF** ($\tau_y = 10^{30}\mathrm{\ Pa}$):
   - Face [-3] ($r = 6258.52\mathrm{\ km}$, sub-lid):
     $w_\mathrm{lid} = 0.0$, $\kappa_h = 0.6788\mathrm{\ m^2/s}$, $j_\mathrm{cond} = 0.0218\mathrm{\ W/m^2}$, $j_\mathrm{conv} = 2.4568 \times 10^5\mathrm{\ W/m^2}$, $j_\mathrm{mix} = 0.0037\mathrm{\ W/m^2}$, $F_\mathrm{total} = 2.4568 \times 10^5\mathrm{\ W/m^2}$.
   - Face [-2] ($r = 6314.55\mathrm{\ km}$, inside lid):
     $w_\mathrm{lid} = 0.8415$, $\kappa_h = 7.618 \times 10^{-3}\mathrm{\ m^2/s}$, $j_\mathrm{cond} = 0.00525\mathrm{\ W/m^2}$, $j_\mathrm{conv} = 24.97\mathrm{\ W/m^2}$, $j_\mathrm{mix} = 0.0\mathrm{\ W/m^2}$, $F_\mathrm{total} = 24.97\mathrm{\ W/m^2}$.
   - Face [-1] ($r = 6371.00\mathrm{\ km}$, surface):
     $w_\mathrm{lid} = 0.999993$, $\kappa_h = 3.169 \times 10^{-7}\mathrm{\ m^2/s}$, $j_\mathrm{cond} = 0.00532\mathrm{\ W/m^2}$, $j_\mathrm{conv} = 9.454 \times 10^{-4}\mathrm{\ W/m^2}$, $j_\mathrm{mix} = 0.0\mathrm{\ W/m^2}$.
     Internal physical heat flux: $j_\mathrm{cond} + j_\mathrm{conv} = 0.00627\mathrm{\ W/m^2}$.
     Imposed surface boundary flux: $F_\mathrm{surf} = 9.6373 \times 10^4\mathrm{\ W/m^2}$.

2. **Yielding ON** ($\tau_y = 10^5\mathrm{\ Pa}$, using current code $\tau_d = 2.65 \times 10^{19}\mathrm{\ Pa}$):
   - Face [-3] ($r = 6258.52\mathrm{\ km}$, sub-lid):
     $w_\mathrm{lid} = 0.0$, $\kappa_h = 0.6588\mathrm{\ m^2/s}$, $j_\mathrm{cond} = 0.0220\mathrm{\ W/m^2}$, $j_\mathrm{conv} = 2.4007 \times 10^5\mathrm{\ W/m^2}$, $j_\mathrm{mix} = 0.0034\mathrm{\ W/m^2}$, $F_\mathrm{total} = 2.4007 \times 10^5\mathrm{\ W/m^2}$.
   - Face [-2] ($r = 6314.55\mathrm{\ km}$, inside yielded lid):
     $w_\mathrm{lid} = 0.9328$, $\kappa_h = 1.6248 \times 10^5\mathrm{\ m^2/s}$, $j_\mathrm{cond} = 0.00203\mathrm{\ W/m^2}$, $j_\mathrm{conv} = 1.7427 \times 10^5\mathrm{\ W/m^2}$, $j_\mathrm{mix} = -0.2919\mathrm{\ W/m^2}$, $F_\mathrm{total} = 1.7427 \times 10^5\mathrm{\ W/m^2}$.
   - Face [-1] ($r = 6371.00\mathrm{\ km}$, surface):
     $w_\mathrm{lid} = 1.0000$, $\kappa_h = 1.7681 \times 10^5\mathrm{\ m^2/s}$, $j_\mathrm{cond} = 0.00247\mathrm{\ W/m^2}$, $j_\mathrm{conv} = 1.7876 \times 10^5\mathrm{\ W/m^2}$, $j_\mathrm{mix} = 0.0\mathrm{\ W/m^2}$.
     Imposed surface boundary flux: $F_\mathrm{surf} = 1.1200 \times 10^5\mathrm{\ W/m^2}$.

### 4.2 Sanity Verification of Lid Conduction ($j_\mathrm{cond} \approx 0.005\mathrm{\ W/m^2}$)

The conductive heat flux of $j_\mathrm{cond} \approx 0.00525\mathrm{\ W/m^2}$ ($5.25\mathrm{\ mW/m^2}$) at Face [-2] appears low compared to mature terrestrial lithosphere ($40 - 80\mathrm{\ mW/m^2}$). We verify the exact numerical and physical components:

1. **Bracketing Staggered Nodes for Face [-2]** ($r_\mathrm{face} = 6314550.51\mathrm{\ m}$):
   - Inner Node (Node [-2]): $r = 6286484.58\mathrm{\ m}$ ($6286.48\mathrm{\ km}$), $T = 1253.96\mathrm{\ K}$
   - Outer Node (Node [-1]): $r = 6342722.40\mathrm{\ m}$ ($6342.72\mathrm{\ km}$), $T = 1180.44\mathrm{\ K}$
   - Node spacing: $\Delta r = r[-1] - r[-2] = 56237.83\mathrm{\ m}$ ($56.24\mathrm{\ km}$)
   - Temperature drop: $\Delta T = T[-1] - T[-2] = -73.53\mathrm{\ K}$
   - Temperature gradient:
     $$\left| \frac{dT}{dr} \right| = \frac{73.53\mathrm{\ K}}{56237.83\mathrm{\ m}} = 1.3074 \times 10^{-3}\mathrm{\ K/m} \quad (1.307\mathrm{\ K/km})$$

2. **Conductive Heat Flux Calculation**:
   - Thermal conductivity: $k = 4.016\mathrm{\ W/(m\ K)}$ (config parameter `thermal_conductivity = 4.0` in `phase_solid` and `phase_liquid`)
   - The conductive heat flux evaluates to:
     $$j_\mathrm{cond} = k \left| \frac{dT}{dr} \right| = 4.016\mathrm{\ W/(m\ K)} \times 1.3074 \times 10^{-3}\mathrm{\ K/m} = 0.00525\mathrm{\ W/m^2} \quad (5.25\mathrm{\ mW/m^2})$$

3. **Physical Explanation**:
   - The 20-node uniform radial grid across a $1000\mathrm{\ km}$ mantle gives coarse cells of thickness $\Delta r \approx 56.24\mathrm{\ km}$.
   - At $t = 0.10\mathrm{\ yr}$, a thin conductive skin has not yet steepened the interior geothermal gradient across the $56\mathrm{\ km}$ cell; the temperature drop between the two uppermost coarse nodes is only $73.53\mathrm{\ K}$, corresponding to a gradient of only $1.31\mathrm{\ K/km}$ (compared to $20 - 30\mathrm{\ K/km}$ in a thin $10\mathrm{\ km}$ terrestrial lid).
   - Conduction through rock ($k \approx 4\mathrm{\ W/(m\ K)}$) over $56.24\mathrm{\ km}$ with $\Delta T = 73.53\mathrm{\ K}$ physically yields $0.00525\mathrm{\ W/m^2}$.
   - The large surface heat flux ($9.64 \times 10^4\mathrm{\ W/m^2}$) is the grey-body radiation boundary condition ($F_\mathrm{surf} = \epsilon \sigma (T_\mathrm{surf}^4 - T_\mathrm{eq}^4)$) radiating directly into space from the hot outermost cell ($T_\mathrm{surf} \approx 1180\mathrm{\ K}$), not conductive throughput across the entire lid thickness.

### 4.3 Origin of the $\sim 10^5\mathrm{\ W/m^2}$ Surface Heat Flux

In `tests/configs/cold_top_lid.toml`, the boundary configuration specifies:
```toml
[boundary_conditions]
outer_boundary_condition = 1
outer_boundary_value = 400.0
emissivity = 1
equilibrium_temperature = 273
```
In Aragog (`src/aragog/solver/entropy_solver.py:2511-2518`):
- `outer_boundary_condition = 1` sets **grey-body radiation**:
  $$F_\mathrm{surf} = \epsilon \sigma (T_\mathrm{surf}^4 - T_\mathrm{eq}^4)$$
  The parameter `outer_boundary_value = 400.0` is not read in mode 1. (Prescribed surface temperature is mode 5).
- Because the surface staggered node temperature is $T_\mathrm{surf} \approx 1142.7\mathrm{\ K}$ (yielding off) and $1186.3\mathrm{\ K}$ (yielding on), the grey-body radiation law evaluates to:
  $$\sigma (1142.7^4 - 273^4) = 96372.68\mathrm{\ W/m^2} \quad (\text{Yielding OFF})$$
  $$\sigma (1186.3^4 - 273^4) = 112001.80\mathrm{\ W/m^2} \quad (\text{Yielding ON})$$
- Conduction through the lid does not carry this heat flux. Internal conductive heat flux at the surface is $j_\mathrm{cond} = 0.0053\mathrm{\ W/m^2}$.
- When yielding is OFF, the stagnant lid suppresses convective eddy transport:
  - $\kappa_h$ at face [-1] drops by $2.14 \times 10^6 \times$ (from $0.68\mathrm{\ m^2/s}$ in the sub-lid to $3.17 \times 10^{-7}\mathrm{\ m^2/s}$).
  - Convective heat flux in the lid drops by $1.89 \times 10^8 \times$ (from $1.79 \times 10^5\mathrm{\ W/m^2}$ to $9.45 \times 10^{-4}\mathrm{\ W/m^2}$).
  - Convective heat flux across face [-2] is only $24.97\mathrm{\ W/m^2}$.
- When yielding is ON, the unphysical driving stress $\tau_d = 2.65 \times 10^{19}\mathrm{\ Pa}$ reduces the lid viscosity to $1.83 \times 10^5\mathrm{\ Pa\ s}$, mobilizing the lid and permitting eddy transport ($j_\mathrm{conv} \approx 1.79 \times 10^5\mathrm{\ W/m^2}$).
- The state effect with the corrected buoyancy stress ($\tau_\mathrm{buoy} = 1.309\mathrm{\ MPa}$) will be simulated after confirmation from the architect and Tim.

---

## 5. Noise Floor Verification across 5 Perturbation Seeds

Measured by applying 1-ULP random sign perturbations, $s_0 \pm \mathrm{spacing}(s_0)$, to the initial entropy profile across 5 seeds:

| Seed | Sign Pattern | $\max |\Delta S|$ [J kg$^{-1}$ K$^{-1}$] | $\max |\Delta T|$ [K] | $\max |\Delta q|$ [W m$^{-2}$] | $\max |\Delta \tau_\mathrm{lid}|$ [Pa] |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 42 | `[-1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1]` | 1.2196e-08 | 1.2665e-09 | 4.3628e+01 | 3.4110e+11 |
| 123 | `[-1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1]` | 1.4593e-09 | 1.5189e-10 | 6.8965e+00 | 6.5941e+08 |
| 2026 | `[-1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1]` | 7.9085e-09 | 8.2127e-10 | 2.9640e+01 | 3.3932e+11 |
| 7777 | `[1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1]` | 1.1241e-08 | 1.1673e-09 | 4.0070e+01 | 3.4073e+11 |
| 99999 | `[1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1]` | 4.0340e-08 | 4.1873e-09 | 1.3404e+02 | 1.7363e+10 |

**Maximum noise floor across 5 seeds**:
- Entropy: $\max |\Delta S| = 4.0340 \times 10^{-8}\mathrm{\ J\ kg^{-1}\ K^{-1}}$
- Temperature: $\max |\Delta T| = 4.1873 \times 10^{-9}\mathrm{\ K}$
- Heat flux: $\max |\Delta q| = 134.04\mathrm{\ W\ m^{-2}}$
- Lid stress: $\max |\Delta \tau_\mathrm{lid}| = 3.4110 \times 10^{11}\mathrm{\ Pa}$

**Fixture tolerances** (set to $\ge 10\times$ noise floor):
- `atol(S) = 1.0e-6 J/kg/K` ($24.8\times$ noise floor)
- `atol(T) = 1.0e-7 K` ($23.9\times$ noise floor)
- `atol(flux) = 2000.0 W/m^2` ($14.9\times$ noise floor)
- `atol(lid_stress) = 5.0e12 Pa` ($14.7\times$ noise floor)

The yielding state change ($\max |\Delta S| = 24.29\mathrm{\ J\ kg^{-1}\ K^{-1}}$) exceeds the measured 5-seed noise floor by $6.02 \times 10^8 \times$.

---

## 6. Documentation Updates Completed

1. **Harmonic Mean Yield Closure**:
   - Described in `docs/Explanations/solid_state_convection.md`, Section 5.
   - Cites implementation file and lines: `src/aragog/rheology_lid.py:344-365`.
   - Obsolete "Two-Branch Regime Switch" text was removed.
   - Split tables in `solid_state_convection.md` and `configuration.md` were unified.

2. **Lid Contrast Coefficient Choice**:
   - Configured as `lid_contrast_coeff = 2.2` in `SolidRheologyParams` (`src/aragog/rheology.py:44`).
   - Rationale: Solomatov (1995) and Tackley (2000) establish $a \approx 2.2$ for Newtonian convection with strongly temperature-dependent viscosity. Foley & Bercovici (2014) use lower values ($a_\mathrm{rh} = 1.3 - 1.82$) to include grain damage softening. Because Step 1 implements diffusion creep without grain damage mechanics, Aragog keeps $a = 2.2$ as the Newtonian baseline.
