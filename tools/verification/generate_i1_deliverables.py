import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Target directories
output_dirs = [
    Path("/Users/timlichtenberg/.shared-agent-state/streams/ssc-step1/data/I1"),
    Path("/Users/timlichtenberg/work/stream-ssc-step1/wt-repairs/data/I1"),
]
for d in output_dirs:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Case 1: Solid Earth Reference State (docs/Explanations/solid_state_convection.md:229)
# ---------------------------------------------------------
rho_earth = 3300.0  # kg/m3
g_earth = 9.81  # m/s2
alpha_earth = 3.0e-5  # 1/K
T_i_earth = 1600.0  # K
T_s_earth = 300.0  # K
DeltaT_earth = T_i_earth - T_s_earth  # 1300 K
d_earth = 2890.0e3  # 2890 km
kappa_earth = 1.0e-6  # m2/s
P_Ti_earth = 3.0e9  # 3 GPa
dT_rh_earth = 68.0  # K
delta_rh_earth = 6.8e3  # 6.8 km
eta_i_earth = 1.0e20  # Pa s
eta_bulk_earth = 1.0e20  # Pa s (solid)
v_i_earth = 7.92e-11  # m/s (2.5 mm/yr)
tau_y_earth_lab = 200.0e6  # 200 MPa (Byerlee lab)
tau_y_earth_model = 50.0e6  # 50 MPa (model threshold)

# Options for Earth
# Current code:
tau_d_earth_curr = eta_i_earth * v_i_earth / delta_rh_earth  # 1.165e6 Pa = 1.165 MPa
# Option (a) consistent viscosity:
tau_d_earth_opt_a = eta_bulk_earth * v_i_earth / delta_rh_earth  # 1.165 MPa
# Option (b) solid velocity:
tau_d_earth_opt_b = eta_i_earth * v_i_earth / delta_rh_earth  # 1.165 MPa

# Option (b') Published stagnant lid velocity (Foley & Bercovici 2014, GJI 199, p. 588, Eq. 28)
# tau_xz = 2 * mu_eff * v_m / d with mu_eff = mu_i
# Ra_eff = rho * g * alpha * DeltaT * d^3 / (kappa * eta_i)
Ra_eff_earth = rho_earth * g_earth * alpha_earth * DeltaT_earth * (d_earth**3) / (kappa_earth * eta_i_earth)
E_act = 300.0e3  # J/mol
R_gas = 8.314  # J/mol/K
theta_earth = (E_act * DeltaT_earth) / (R_gas * (T_i_earth**2))
C4 = 0.125
a_rh_fb = 1.3
vm_fb_earth = (kappa_earth / d_earth) * C4 * ((Ra_eff_earth * a_rh_fb / theta_earth)**(2.0 / 3.0))
# Factor of 2 from FB2014 Eq. 28 / p. 588:
tau_fb_d_earth = 2.0 * eta_i_earth * vm_fb_earth / d_earth

# Also evaluate variant with reduced convective sublayer DeltaT_sub = 130 K:
DeltaT_sub_earth = 130.0  # K
Ra_eff_earth_sub = rho_earth * g_earth * alpha_earth * DeltaT_sub_earth * (d_earth**3) / (kappa_earth * eta_i_earth)
vm_fb_earth_sub = (kappa_earth / d_earth) * C4 * ((Ra_eff_earth_sub * a_rh_fb / theta_earth)**(2.0 / 3.0))
tau_fb_d_earth_sub = 2.0 * eta_i_earth * vm_fb_earth_sub / d_earth

# Option (b') Solomatov & Moresi 2000 (1/2 exponent):
# ui = 0.38 * (Ra / theta)^(1/2) * (kappa / d)
vm_sm_earth = (kappa_earth / d_earth) * 0.38 * ((Ra_eff_earth / theta_earth)**0.5)
tau_sm_d_earth = 2.0 * eta_i_earth * vm_sm_earth / d_earth

vm_sm_earth_sub = (kappa_earth / d_earth) * 0.38 * ((Ra_eff_earth_sub / theta_earth)**0.5)
tau_sm_d_earth_sub = 2.0 * eta_i_earth * vm_sm_earth_sub / d_earth

# Option (c) Buoyancy stress:
tau_buoy_earth = rho_earth * g_earth * alpha_earth * dT_rh_earth * delta_rh_earth  # 0.449 MPa

# ---------------------------------------------------------
# Case 2: Cold-Top-Lid Fixture (tests/configs/cold_top_lid.toml)
# ---------------------------------------------------------
rho_fixture = 3157.36  # kg/m3
g_fixture = 9.81  # m/s2
alpha_fixture = 4.180e-5  # 1/K
T_i_fixture = 1998.19  # K
T_s_fixture = 400.0  # K
DeltaT_fixture = T_i_fixture - T_s_fixture  # 1598.19 K
d_fixture = 1000.0e3  # 1000 km
kappa_fixture = 1.0e-6  # m2/s
P_Ti_fixture = 4.40e9  # 4.40 GPa
dT_rh_fixture = 102.33  # K
delta_rh_fixture = 9878.40  # m
eta_i_fixture = 4.8595e19  # Pa s
eta_bulk_fixture = 3.108e7  # Pa s (mushy sub-lid)
v_i_fixture = 5396.29  # m/s (inertial MLT velocity)
v_solid_fixture = 3.451e-9  # m/s (solid-scaled velocity)
tau_y_fixture = 1.0e5  # 1e5 Pa = 0.1 MPa

# Options for Fixture
# Current code:
tau_d_fixture_curr = eta_i_fixture * v_i_fixture / delta_rh_fixture  # 2.655e19 Pa
# Option (a) consistent viscosity:
tau_d_fixture_opt_a = eta_bulk_fixture * v_i_fixture / delta_rh_fixture  # 1.698e7 Pa = 16.98 MPa
# Option (b) solid velocity:
tau_d_fixture_opt_b = eta_i_fixture * v_solid_fixture / delta_rh_fixture  # 1.698e7 Pa = 16.98 MPa

# Option (b') with solid viscosity eta_i:
Ra_eff_fixture_solid = rho_fixture * g_fixture * alpha_fixture * DeltaT_fixture * (d_fixture**3) / (kappa_fixture * eta_i_fixture)
theta_fixture = (E_act * DeltaT_fixture) / (R_gas * (T_i_fixture**2))
vm_fb_fixture_solid = (kappa_fixture / d_fixture) * C4 * ((Ra_eff_fixture_solid * a_rh_fb / theta_fixture)**(2.0 / 3.0))
tau_fb_d_fixture_solid = 2.0 * eta_i_fixture * vm_fb_fixture_solid / d_fixture

vm_sm_fixture_solid = (kappa_fixture / d_fixture) * 0.38 * ((Ra_eff_fixture_solid / theta_fixture)**0.5)
tau_sm_d_fixture_solid = 2.0 * eta_i_fixture * vm_sm_fixture_solid / d_fixture

# Option (b') with mushy bulk viscosity eta_bulk (Ruling 97(2)):
Ra_eff_fixture_mush = rho_fixture * g_fixture * alpha_fixture * DeltaT_fixture * (d_fixture**3) / (kappa_fixture * eta_bulk_fixture)
vm_fb_fixture_mush = (kappa_fixture / d_fixture) * C4 * ((Ra_eff_fixture_mush * a_rh_fb / theta_fixture)**(2.0 / 3.0))
tau_fb_d_fixture_mush = 2.0 * eta_bulk_fixture * vm_fb_fixture_mush / d_fixture

# Option (c) Buoyancy stress:
tau_buoy_fixture = rho_fixture * g_fixture * alpha_fixture * dT_rh_fixture * delta_rh_fixture  # 1.309 MPa

# Build Comparison Table with all requested input parameters
comparison_rows = [
    {
        "case": "Solid Earth Example",
        "option": "Current Code: eta_i * v_i / delta_rh",
        "rho": f"{rho_earth:.1f}",
        "g": f"{g_earth:.2f}",
        "alpha": f"{alpha_earth:.2e}",
        "DeltaT_rh": f"{dT_rh_earth:.1f}",
        "delta_rh": f"{delta_rh_earth:.1f}",
        "d": f"{d_earth:.2e}",
        "kappa": f"{kappa_earth:.2e}",
        "theta": f"{theta_earth:.2f}",
        "Ra_eff": f"{Ra_eff_earth:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_earth:.2e}",
        "eta_i": f"{eta_i_earth:.2e}",
        "v_conv": f"{v_i_earth:.2e}",
        "tau_d_Pa": f"{tau_d_earth_curr:.4e}",
        "tau_d_MPa": f"{tau_d_earth_curr/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_earth_model:.1e}",
        "yields": "NO" if tau_d_earth_curr < tau_y_earth_model else "YES",
    },
    {
        "case": "Solid Earth Example",
        "option": "(a) Consistent Viscosity: eta_bulk * v_i / delta_rh",
        "rho": f"{rho_earth:.1f}",
        "g": f"{g_earth:.2f}",
        "alpha": f"{alpha_earth:.2e}",
        "DeltaT_rh": f"{dT_rh_earth:.1f}",
        "delta_rh": f"{delta_rh_earth:.1f}",
        "d": f"{d_earth:.2e}",
        "kappa": f"{kappa_earth:.2e}",
        "theta": f"{theta_earth:.2f}",
        "Ra_eff": f"{Ra_eff_earth:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_earth:.2e}",
        "eta_i": f"{eta_i_earth:.2e}",
        "v_conv": f"{v_i_earth:.2e}",
        "tau_d_Pa": f"{tau_d_earth_opt_a:.4e}",
        "tau_d_MPa": f"{tau_d_earth_opt_a/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_earth_model:.1e}",
        "yields": "NO",
    },
    {
        "case": "Solid Earth Example",
        "option": "(b) Solid Velocity: eta_i * v_solid / delta_rh",
        "rho": f"{rho_earth:.1f}",
        "g": f"{g_earth:.2f}",
        "alpha": f"{alpha_earth:.2e}",
        "DeltaT_rh": f"{dT_rh_earth:.1f}",
        "delta_rh": f"{delta_rh_earth:.1f}",
        "d": f"{d_earth:.2e}",
        "kappa": f"{kappa_earth:.2e}",
        "theta": f"{theta_earth:.2f}",
        "Ra_eff": f"{Ra_eff_earth:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_earth:.2e}",
        "eta_i": f"{eta_i_earth:.2e}",
        "v_conv": f"{v_i_earth:.2e}",
        "tau_d_Pa": f"{tau_d_earth_opt_b:.4e}",
        "tau_d_MPa": f"{tau_d_earth_opt_b/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_earth_model:.1e}",
        "yields": "NO",
    },
    {
        "case": "Solid Earth Example",
        "option": "(b') FB2014 (2/3, whole mantle): 2*eta_i*v_m/d",
        "rho": f"{rho_earth:.1f}",
        "g": f"{g_earth:.2f}",
        "alpha": f"{alpha_earth:.2e}",
        "DeltaT_rh": f"{dT_rh_earth:.1f}",
        "delta_rh": f"{delta_rh_earth:.1f}",
        "d": f"{d_earth:.2e}",
        "kappa": f"{kappa_earth:.2e}",
        "theta": f"{theta_earth:.2f}",
        "Ra_eff": f"{Ra_eff_earth:.2e}",
        "arh": f"{a_rh_fb:.2f}",
        "eta_bulk": f"{eta_bulk_earth:.2e}",
        "eta_i": f"{eta_i_earth:.2e}",
        "v_conv": f"{vm_fb_earth:.2e}",
        "tau_d_Pa": f"{tau_fb_d_earth:.4e}",
        "tau_d_MPa": f"{tau_fb_d_earth/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_earth_model:.1e}",
        "yields": "NO",
    },
    {
        "case": "Solid Earth Example",
        "option": "(b') FB2014 (2/3, sublayer dT=130K): 2*eta_i*v_m/d",
        "rho": f"{rho_earth:.1f}",
        "g": f"{g_earth:.2f}",
        "alpha": f"{alpha_earth:.2e}",
        "DeltaT_rh": f"{dT_rh_earth:.1f}",
        "delta_rh": f"{delta_rh_earth:.1f}",
        "d": f"{d_earth:.2e}",
        "kappa": f"{kappa_earth:.2e}",
        "theta": f"{theta_earth:.2f}",
        "Ra_eff": f"{Ra_eff_earth_sub:.2e}",
        "arh": f"{a_rh_fb:.2f}",
        "eta_bulk": f"{eta_bulk_earth:.2e}",
        "eta_i": f"{eta_i_earth:.2e}",
        "v_conv": f"{vm_fb_earth_sub:.2e}",
        "tau_d_Pa": f"{tau_fb_d_earth_sub:.4e}",
        "tau_d_MPa": f"{tau_fb_d_earth_sub/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_earth_model:.1e}",
        "yields": "NO",
    },
    {
        "case": "Solid Earth Example",
        "option": "(b') SM2000 (1/2, whole mantle): 2*eta_i*v_m/d",
        "rho": f"{rho_earth:.1f}",
        "g": f"{g_earth:.2f}",
        "alpha": f"{alpha_earth:.2e}",
        "DeltaT_rh": f"{dT_rh_earth:.1f}",
        "delta_rh": f"{delta_rh_earth:.1f}",
        "d": f"{d_earth:.2e}",
        "kappa": f"{kappa_earth:.2e}",
        "theta": f"{theta_earth:.2f}",
        "Ra_eff": f"{Ra_eff_earth:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_earth:.2e}",
        "eta_i": f"{eta_i_earth:.2e}",
        "v_conv": f"{vm_sm_earth:.2e}",
        "tau_d_Pa": f"{tau_sm_d_earth:.4e}",
        "tau_d_MPa": f"{tau_sm_d_earth/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_earth_model:.1e}",
        "yields": "NO",
    },
    {
        "case": "Solid Earth Example",
        "option": "(b') SM2000 (1/2, sublayer dT=130K): 2*eta_i*v_m/d",
        "rho": f"{rho_earth:.1f}",
        "g": f"{g_earth:.2f}",
        "alpha": f"{alpha_earth:.2e}",
        "DeltaT_rh": f"{dT_rh_earth:.1f}",
        "delta_rh": f"{delta_rh_earth:.1f}",
        "d": f"{d_earth:.2e}",
        "kappa": f"{kappa_earth:.2e}",
        "theta": f"{theta_earth:.2f}",
        "Ra_eff": f"{Ra_eff_earth_sub:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_earth:.2e}",
        "eta_i": f"{eta_i_earth:.2e}",
        "v_conv": f"{vm_sm_earth_sub:.2e}",
        "tau_d_Pa": f"{tau_sm_d_earth_sub:.4e}",
        "tau_d_MPa": f"{tau_sm_d_earth_sub/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_earth_model:.1e}",
        "yields": "NO",
    },
    {
        "case": "Solid Earth Example",
        "option": "(c) Buoyancy Stress: rho * g * alpha * DeltaT_rh * delta_rh",
        "rho": f"{rho_earth:.1f}",
        "g": f"{g_earth:.2f}",
        "alpha": f"{alpha_earth:.2e}",
        "DeltaT_rh": f"{dT_rh_earth:.1f}",
        "delta_rh": f"{delta_rh_earth:.1f}",
        "d": f"{d_earth:.2e}",
        "kappa": f"{kappa_earth:.2e}",
        "theta": f"{theta_earth:.2f}",
        "Ra_eff": f"{Ra_eff_earth:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_earth:.2e}",
        "eta_i": f"{eta_i_earth:.2e}",
        "v_conv": "N/A",
        "tau_d_Pa": f"{tau_buoy_earth:.4e}",
        "tau_d_MPa": f"{tau_buoy_earth/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_earth_model:.1e}",
        "yields": "NO",
    },
    {
        "case": "Cold Top Lid Fixture",
        "option": "Current Code: eta_i * v_i / delta_rh",
        "rho": f"{rho_fixture:.1f}",
        "g": f"{g_fixture:.2f}",
        "alpha": f"{alpha_fixture:.2e}",
        "DeltaT_rh": f"{dT_rh_fixture:.1f}",
        "delta_rh": f"{delta_rh_fixture:.1f}",
        "d": f"{d_fixture:.2e}",
        "kappa": f"{kappa_fixture:.2e}",
        "theta": f"{theta_fixture:.2f}",
        "Ra_eff": f"{Ra_eff_fixture_solid:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_fixture:.2e}",
        "eta_i": f"{eta_i_fixture:.2e}",
        "v_conv": f"{v_i_fixture:.2f}",
        "tau_d_Pa": f"{tau_d_fixture_curr:.4e}",
        "tau_d_MPa": f"{tau_d_fixture_curr/1e6:.4e}",
        "tau_y_Pa": f"{tau_y_fixture:.1e}",
        "yields": "YES (unphysical)",
    },
    {
        "case": "Cold Top Lid Fixture",
        "option": "(a) Consistent Viscosity: eta_bulk * v_i / delta_rh",
        "rho": f"{rho_fixture:.1f}",
        "g": f"{g_fixture:.2f}",
        "alpha": f"{alpha_fixture:.2e}",
        "DeltaT_rh": f"{dT_rh_fixture:.1f}",
        "delta_rh": f"{delta_rh_fixture:.1f}",
        "d": f"{d_fixture:.2e}",
        "kappa": f"{kappa_fixture:.2e}",
        "theta": f"{theta_fixture:.2f}",
        "Ra_eff": f"{Ra_eff_fixture_solid:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_fixture:.2e}",
        "eta_i": f"{eta_i_fixture:.2e}",
        "v_conv": f"{v_i_fixture:.2f}",
        "tau_d_Pa": f"{tau_d_fixture_opt_a:.4e}",
        "tau_d_MPa": f"{tau_d_fixture_opt_a/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_fixture:.1e}",
        "yields": "YES",
    },
    {
        "case": "Cold Top Lid Fixture",
        "option": "(b) Solid Velocity: eta_i * v_solid / delta_rh",
        "rho": f"{rho_fixture:.1f}",
        "g": f"{g_fixture:.2f}",
        "alpha": f"{alpha_fixture:.2e}",
        "DeltaT_rh": f"{dT_rh_fixture:.1f}",
        "delta_rh": f"{delta_rh_fixture:.1f}",
        "d": f"{d_fixture:.2e}",
        "kappa": f"{kappa_fixture:.2e}",
        "theta": f"{theta_fixture:.2f}",
        "Ra_eff": f"{Ra_eff_fixture_solid:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_fixture:.2e}",
        "eta_i": f"{eta_i_fixture:.2e}",
        "v_conv": f"{v_solid_fixture:.2e}",
        "tau_d_Pa": f"{tau_d_fixture_opt_b:.4e}",
        "tau_d_MPa": f"{tau_d_fixture_opt_b/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_fixture:.1e}",
        "yields": "YES",
    },
    {
        "case": "Cold Top Lid Fixture",
        "option": "(b') FB2014 (solid mu_i=eta_i): 2*eta_i*v_m/d",
        "rho": f"{rho_fixture:.1f}",
        "g": f"{g_fixture:.2f}",
        "alpha": f"{alpha_fixture:.2e}",
        "DeltaT_rh": f"{dT_rh_fixture:.1f}",
        "delta_rh": f"{delta_rh_fixture:.1f}",
        "d": f"{d_fixture:.2e}",
        "kappa": f"{kappa_fixture:.2e}",
        "theta": f"{theta_fixture:.2f}",
        "Ra_eff": f"{Ra_eff_fixture_solid:.2e}",
        "arh": f"{a_rh_fb:.2f}",
        "eta_bulk": f"{eta_bulk_fixture:.2e}",
        "eta_i": f"{eta_i_fixture:.2e}",
        "v_conv": f"{vm_fb_fixture_solid:.2e}",
        "tau_d_Pa": f"{tau_fb_d_fixture_solid:.4e}",
        "tau_d_MPa": f"{tau_fb_d_fixture_solid/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_fixture:.1e}",
        "yields": "YES (at tau_y=0.1 MPa; NO at tau_y >= 0.5 MPa)",
    },
    {
        "case": "Cold Top Lid Fixture",
        "option": "(b') FB2014 (mushy mu_i=eta_bulk): 2*eta_bulk*v_m/d",
        "rho": f"{rho_fixture:.1f}",
        "g": f"{g_fixture:.2f}",
        "alpha": f"{alpha_fixture:.2e}",
        "DeltaT_rh": f"{dT_rh_fixture:.1f}",
        "delta_rh": f"{delta_rh_fixture:.1f}",
        "d": f"{d_fixture:.2e}",
        "kappa": f"{kappa_fixture:.2e}",
        "theta": f"{theta_fixture:.2f}",
        "Ra_eff": f"{Ra_eff_fixture_mush:.2e}",
        "arh": f"{a_rh_fb:.2f}",
        "eta_bulk": f"{eta_bulk_fixture:.2e}",
        "eta_i": f"{eta_i_fixture:.2e}",
        "v_conv": f"{vm_fb_fixture_mush:.2e}",
        "tau_d_Pa": f"{tau_fb_d_fixture_mush:.4e}",
        "tau_d_MPa": f"{tau_fb_d_fixture_mush/1e6:.4e}",
        "tau_y_Pa": f"{tau_y_fixture:.1e}",
        "yields": "NO (Ra=6.7e19 >> FB2014 range)",
    },
    {
        "case": "Cold Top Lid Fixture",
        "option": "(b') SM2000 (solid mu_i=eta_i, 1/2): 2*eta_i*v_m/d",
        "rho": f"{rho_fixture:.1f}",
        "g": f"{g_fixture:.2f}",
        "alpha": f"{alpha_fixture:.2e}",
        "DeltaT_rh": f"{dT_rh_fixture:.1f}",
        "delta_rh": f"{delta_rh_fixture:.1f}",
        "d": f"{d_fixture:.2e}",
        "kappa": f"{kappa_fixture:.2e}",
        "theta": f"{theta_fixture:.2f}",
        "Ra_eff": f"{Ra_eff_fixture_solid:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_fixture:.2e}",
        "eta_i": f"{eta_i_fixture:.2e}",
        "v_conv": f"{vm_sm_fixture_solid:.2e}",
        "tau_d_Pa": f"{tau_sm_d_fixture_solid:.4e}",
        "tau_d_MPa": f"{tau_sm_d_fixture_solid/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_fixture:.1e}",
        "yields": "NO",
    },
    {
        "case": "Cold Top Lid Fixture",
        "option": "(c) Buoyancy Stress: rho * g * alpha * DeltaT_rh * delta_rh",
        "rho": f"{rho_fixture:.1f}",
        "g": f"{g_fixture:.2f}",
        "alpha": f"{alpha_fixture:.2e}",
        "DeltaT_rh": f"{dT_rh_fixture:.1f}",
        "delta_rh": f"{delta_rh_fixture:.1f}",
        "d": f"{d_fixture:.2e}",
        "kappa": f"{kappa_fixture:.2e}",
        "theta": f"{theta_fixture:.2f}",
        "Ra_eff": f"{Ra_eff_fixture_solid:.2e}",
        "arh": "N/A",
        "eta_bulk": f"{eta_bulk_fixture:.2e}",
        "eta_i": f"{eta_i_fixture:.2e}",
        "v_conv": "N/A",
        "tau_d_Pa": f"{tau_buoy_fixture:.4e}",
        "tau_d_MPa": f"{tau_buoy_fixture/1e6:.4f}",
        "tau_y_Pa": f"{tau_y_fixture:.1e}",
        "yields": "YES (at tau_y=0.1 MPa; NO at tau_y >= 1.5 MPa)",
    },
]

for d in output_dirs:
    with open(d / "tau_d_comparison_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

print("Saved tau_d_comparison_table.csv")

# ---------------------------------------------------------
# Sweep of sub-lid layer: eta_bulk from 1e2 to 1e21 Pa s (mush to solid)
# ---------------------------------------------------------
log_eta_vals = np.linspace(2.0, 21.0, 100)
eta_bulk_sweep = 10.0**log_eta_vals

sweep_rows = []
tau_curr_list = []
tau_opt_a_list = []
tau_opt_b_list = []
tau_opt_b_prime_fb_solid_list = []
tau_opt_b_prime_fb_mush_list = []
tau_opt_b_prime_sm_solid_list = []
tau_opt_c_list = []

for eta_b in eta_bulk_sweep:
    v_inertial = 5396.2873
    v_visc = 5396.2873 * (3.108e7 / eta_b)
    v_i_eff = v_inertial * v_visc / (v_inertial + v_visc)

    # Current code:
    t_curr = eta_i_fixture * v_i_eff / delta_rh_fixture
    # Option (a):
    t_opt_a = eta_b * v_i_eff / delta_rh_fixture
    # Option (b): v_solid uses eta_i
    v_solid_calc = v_inertial * (5396.2873 * (3.108e7 / eta_i_fixture)) / (v_inertial + 5396.2873 * (3.108e7 / eta_i_fixture))
    t_opt_b = eta_i_fixture * v_solid_calc / delta_rh_fixture
    # Option (b') FB2014 solid mu_i = eta_i (with factor 2):
    t_fb_solid = tau_fb_d_fixture_solid
    # Option (b') FB2014 mushy mu_i = eta_b (with factor 2):
    Ra_eff_sw = rho_fixture * g_fixture * alpha_fixture * DeltaT_fixture * (d_fixture**3) / (kappa_fixture * eta_b)
    vm_fb_sw = (kappa_fixture / d_fixture) * C4 * ((Ra_eff_sw * a_rh_fb / theta_fixture)**(2.0 / 3.0))
    t_fb_mush = 2.0 * eta_b * vm_fb_sw / d_fixture
    # Option (b') SM2000 solid:
    t_sm_solid = tau_sm_d_fixture_solid
    # Option (c):
    t_opt_c = tau_buoy_fixture

    tau_curr_list.append(t_curr)
    tau_opt_a_list.append(t_opt_a)
    tau_opt_b_list.append(t_opt_b)
    tau_opt_b_prime_fb_solid_list.append(t_fb_solid)
    tau_opt_b_prime_fb_mush_list.append(t_fb_mush)
    tau_opt_b_prime_sm_solid_list.append(t_sm_solid)
    tau_opt_c_list.append(t_opt_c)

    sweep_rows.append({
        "eta_bulk_Pa_s": f"{eta_b:.4e}",
        "log10_eta_bulk": f"{np.log10(eta_b):.2f}",
        "v_i_m_s": f"{v_i_eff:.4e}",
        "tau_current_Pa": f"{t_curr:.4e}",
        "tau_option_a_Pa": f"{t_opt_a:.4e}",
        "tau_option_b_Pa": f"{t_opt_b:.4e}",
        "tau_option_b_prime_fb_solid_Pa": f"{t_fb_solid:.4e}",
        "tau_option_b_prime_fb_mush_Pa": f"{t_fb_mush:.4e}",
        "tau_option_b_prime_sm_solid_Pa": f"{t_sm_solid:.4e}",
        "tau_option_c_buoy_Pa": f"{t_opt_c:.4e}",
    })

for d in output_dirs:
    with open(d / "tau_d_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

print("Saved tau_d_sweep.csv")

# ---------------------------------------------------------
# Plot: tau_d vs eta_bulk with tau_y marked
# ---------------------------------------------------------
plt.figure(figsize=(9.5, 6.5), dpi=300)
plt.loglog(eta_bulk_sweep, tau_curr_list, 'r-', linewidth=2.5, label=r'Current Code: $\tau_d = \eta_i v_i / \delta_\mathrm{rh}$')
plt.loglog(eta_bulk_sweep, tau_opt_a_list, 'g--', linewidth=2.0, label=r'Option (a): $\tau_d = \eta_\mathrm{bulk} v_i / \delta_\mathrm{rh}$')
plt.loglog(eta_bulk_sweep, tau_opt_b_list, 'm-.', linewidth=2.0, label=r'Option (b): $\tau_d = \eta_i v_{i,\mathrm{solid}} / \delta_\mathrm{rh}$')
plt.loglog(eta_bulk_sweep, tau_opt_c_list, 'b-', linewidth=2.5, label=r'Option (c) Buoyancy: $\tau_\mathrm{buoy} = \rho g \alpha \Delta T_\mathrm{rh} \delta_\mathrm{rh}$')
plt.loglog(eta_bulk_sweep, tau_opt_b_prime_fb_solid_list, 'c:', linewidth=2.0, label=r'Option (b$^\prime$) FB2014 ($\mu_i = \eta_i$, $2\mu_i v_m/d$)')
plt.loglog(eta_bulk_sweep, tau_opt_b_prime_fb_mush_list, color='darkorange', linestyle=':', linewidth=2.0, label=r'Option (b$^\prime$) FB2014 mushy ($\mu_i = \eta_\mathrm{bulk}$, $2\mu_i v_m/d$)')
plt.loglog(eta_bulk_sweep, tau_opt_b_prime_sm_solid_list, color='purple', linestyle=':', linewidth=2.0, label=r'Option (b$^\prime$) SM2000 ($\mu_i = \eta_i$, $v_m \propto \mathrm{Ra}^{1/2}$)')

# Reference Yield Stresses
plt.axhline(1.0e5, color='gray', linestyle='--', alpha=0.7, label=r'$\tau_y = 10^5$ Pa (test fixture)')
plt.axhline(1.0e6, color='gray', linestyle=':', alpha=0.7, label=r'$\tau_y = 10^6$ Pa (1 MPa model)')
plt.axhline(5.0e7, color='black', linestyle='-.', alpha=0.7, label=r'$\tau_y = 50$ MPa (pseudoplastic model threshold)')
plt.axhline(2.0e8, color='black', linestyle='-', alpha=0.7, label=r'$\tau_y = 200$ MPa (laboratory rock strength, Byerlee 1978)')

plt.xlabel(r'Sub-Lid Interior Bulk Viscosity $\eta_\mathrm{bulk}$ [Pa s]', fontsize=12)
plt.ylabel(r'Lid Driving Stress $\tau_d$ [Pa]', fontsize=12)
plt.title(r'Lid Driving Stress Closures vs Interior Bulk Viscosity', fontsize=13)
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.ylim(1.0e1, 1.0e21)
plt.xlim(1.0e2, 1.0e21)
plt.legend(loc='lower left', fontsize=8, framealpha=0.9)
plt.tight_layout()

for d in output_dirs:
    plt.savefig(d / "tau_d_vs_eta_bulk.png")

print("Saved tau_d_vs_eta_bulk.png")
