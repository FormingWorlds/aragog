"""B5: NumPy vs JAX parity verification across 20 random states each.

Verifies:
1. BC 6: `_apply_surface_bc` in `aragog.jax.solver` vs numpy surface block.
2. CMB law: `_cmb_law_flux` in `aragog.jax.solver` vs numpy `_cmb_law_flux`.
3. MLT slope factor: `viscous_mixing_length_factor` (numpy vs JAX).
4. Mesh arrays: `MeshArrays.from_numpy_mesh` vs numpy `Mesh`.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from aragog.cmb_boundary_layer import cmb_flux
from aragog.eos.entropy import EntropyEOS
from aragog.jax.eos import EntropyEOS_JAX
from aragog.jax.phase import MeshArrays
from aragog.jax.solver import BoundaryParams, SIGMA_SB, _apply_surface_bc, _cmb_law_flux
from aragog.mesh import Mesh
from aragog.parser import Parameters
from aragog.rheology import viscous_mixing_length_factor
from aragog.solver.entropy_solver import EntropySolver
from aragog.surface_skin import skin_temperature, solid_weight, table_edge_factor
import aragog


def verify_bc6_surface_flux(n_states: int = 20, seed: int = 42) -> float:
    """Verify parity of BC 6 skin surface flux between NumPy and JAX."""
    rng = np.random.default_rng(seed)
    max_rel_diff = 0.0

    for _ in range(n_states):
        T_basic_top = float(rng.uniform(800.0, 3500.0))
        T_top = float(rng.uniform(800.0, 3500.0))
        k_top = float(rng.uniform(2.0, 10.0))
        dr_half = float(rng.uniform(500.0, 20000.0))
        phi_top = float(rng.uniform(0.0, 1.0))
        phi_rheo = float(rng.uniform(0.3, 0.5))
        emiss = float(rng.uniform(0.6, 1.0))
        T_eq = float(rng.uniform(100.0, 1500.0))
        S_top = float(rng.uniform(2500.0, 3800.0))
        S_edge = 2000.0

        # NumPy
        G = k_top / dr_half
        s = float(solid_weight(phi_top, phi_rheo, xp=np))
        T_s_np = float(skin_temperature(T_top, G, emiss, T_eq, SIGMA_SB, xp=np))
        F_grey_np = emiss * SIGMA_SB * (T_basic_top**4 - T_eq**4)
        F_cond_np = G * (T_top - T_s_np)
        F_surf_np = (1.0 - s) * F_grey_np + s * F_cond_np
        if F_surf_np > 0.0:
            F_surf_np *= float(table_edge_factor(S_top, S_edge, xp=np))

        # JAX
        class DummyMesh:
            radii_basic = jnp.array([6.371e6 - 2.0 * dr_half, 6.371e6])

        class DummyPhaseStag:
            temperature = jnp.array([T_top])
            thermal_conductivity = jnp.array([k_top])
            melt_fraction = jnp.array([phi_top])

        bc = BoundaryParams(
            outer_bc_type=6,
            outer_bc_value=0.0,
            emissivity=emiss,
            T_eq=T_eq,
            inner_bc_type=0,
            inner_bc_value=0.0,
            core_density=8000.0,
            core_heat_capacity=1000.0,
            tfac_core_avg=1.0,
            phi_rheo=phi_rheo,
            table_edge_cutoff=True,
            S_table_edge=S_edge,
        )
        heat_flux = jnp.zeros(2)
        phase_basic_T = jnp.array([1500.0, T_basic_top])
        hf_jax = _apply_surface_bc(
            heat_flux,
            bc,
            phase_basic_T,
            DummyPhaseStag(),
            S_top=jnp.array(S_top),
            mesh=DummyMesh(),
        )
        F_surf_jax = float(hf_jax[-1])

        rel_diff = abs(F_surf_jax - F_surf_np) / abs(F_surf_np)
        if rel_diff > max_rel_diff:
            max_rel_diff = rel_diff

    return max_rel_diff


def verify_cmb_law_flux(n_states: int = 20, seed: int = 42) -> float:
    """Verify parity of CMB flux law between NumPy and JAX solvers."""
    eos_dir = Path(os.environ['ARAGOG_TEST_EOS_DIR'])
    eos_np = EntropyEOS(eos_dir)
    eos_jax = EntropyEOS_JAX(eos_dir)

    cfg = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'
    p = Parameters.from_file(cfg)
    p.mesh.number_of_nodes = 50
    p.boundary_conditions.cmb_flux_law = 'deschamps_sotin_2000'
    p.boundary_conditions.inner_boundary_condition = 3
    p.boundary_conditions.inner_boundary_value = 3500.0
    p.boundary_conditions.outer_boundary_condition = 5
    p.boundary_conditions.outer_boundary_value = 1600.0

    solver = EntropySolver(p, eos_np)
    solver.initialize()
    solver.set_initial_entropy(3000.0)
    rs = np.asarray(solver.evaluator.mesh.staggered.radii).ravel()

    rng = np.random.default_rng(seed)
    max_rel_diff = 0.0

    for _ in range(n_states):
        S = 2800.0 + rng.uniform(-100.0, 100.0, len(rs))
        solver.state.update(S, 0.0)

        q_np = solver._cmb_law_flux(S)

        ma = MeshArrays.from_numpy_mesh(solver.evaluator.mesh)
        ps = solver.state.phase_staggered
        phase_stag_jax = type(
            'PhaseStag',
            (),
            {
                'density': jnp.array(np.asarray(ps.density()).ravel()),
                'viscosity': jnp.array(np.asarray(ps.viscosity()).ravel()),
                'thermal_conductivity': jnp.array(np.asarray(ps.thermal_conductivity()).ravel()),
                'heat_capacity': jnp.array(np.asarray(ps.heat_capacity()).ravel()),
                'thermal_expansivity': jnp.array(np.asarray(ps.thermal_expansivity()).ravel()),
                'temperature': jnp.array(np.asarray(ps.temperature()).ravel()),
            },
        )()

        bc = BoundaryParams(
            outer_bc_type=5,
            outer_bc_value=1600.0,
            emissivity=1.0,
            T_eq=300.0,
            inner_bc_type=3,
            inner_bc_value=3500.0,
            core_density=8000.0,
            core_heat_capacity=1000.0,
            tfac_core_avg=1.0,
            cmb_flux_law=True,
            cmb_law_interior=jnp.array(solver._cmb_law_interior, dtype=jnp.float64),
        )

        q_jax = _cmb_law_flux(
            bc,
            ma,
            eos_jax,
            phase_stag_jax,
            jnp.array(S),
            1600.0,
            T_c=3500.0,
            face=0,
        )

        rel_diff = abs(float(q_jax) - q_np) / abs(q_np)
        if rel_diff > max_rel_diff:
            max_rel_diff = rel_diff

    return max_rel_diff


def verify_mlt_slope_factor(n_states: int = 20, seed: int = 42) -> float:
    """Verify parity of MLT slope factor q between NumPy and JAX."""
    rng = np.random.default_rng(seed)
    max_rel_diff = 0.0

    for _ in range(n_states):
        n = int(rng.integers(20, 200))
        r_in = float(rng.uniform(2.5e6, 3.5e6))
        r_out = float(rng.uniform(5.5e6, 6.5e6))
        radii = np.linspace(r_in, r_out, n)
        mixing_length = np.sin(np.pi * (radii - r_in) / (r_out - r_in)) * 1e5
        mixing_length[0] = 0.0
        mixing_length[-1] = 0.0
        top_slope = float(rng.uniform(0.1, 0.5))
        bottom_slope = float(rng.uniform(0.1, 0.5))

        q_np = viscous_mixing_length_factor(
            radii, r_in, r_out, mixing_length, top_slope, bottom_slope, xp=np
        )
        q_jax = viscous_mixing_length_factor(
            jnp.array(radii),
            r_in,
            r_out,
            jnp.array(mixing_length),
            top_slope,
            bottom_slope,
            xp=jnp,
        )
        rel_diff = float(np.max(np.abs(np.array(q_jax) - q_np) / np.abs(q_np)))
        if rel_diff > max_rel_diff:
            max_rel_diff = rel_diff

    return max_rel_diff


def verify_mesh_arrays(n_states: int = 20, seed: int = 42) -> float:
    """Verify parity between NumPy Mesh and JAX MeshArrays."""
    cfg = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'
    rng = np.random.default_rng(seed)
    max_rel_diff = 0.0

    for _ in range(n_states):
        n = int(rng.integers(30, 150))
        p = Parameters.from_file(cfg)
        p.mesh.number_of_nodes = n
        span = p.mesh.outer_radius - p.mesh.inner_radius
        max_h = 0.8 * span / (n - 1)

        top_candidates = [0.0, 500.0, 1000.0, 2000.0]
        top_cell = float(rng.choice([c for c in top_candidates if c < max_h]))

        bottom_candidates = [0.0, 500.0, 1000.0, 2000.0]
        bottom_cell = float(rng.choice([c for c in bottom_candidates if c < max_h]))

        mass_coord = bool(rng.choice([True, False]))

        p.mesh.mass_coordinates = mass_coord
        p.mesh.surface_cell_thickness = top_cell
        p.mesh.cmb_cell_thickness = bottom_cell

        m = Mesh(p)
        ma = MeshArrays.from_numpy_mesh(m)

        checks = [
            (np.asarray(m.basic.radii).ravel(), np.array(ma.radii_basic)),
            (np.asarray(m.staggered.radii).ravel(), np.array(ma.radii_stag)),
            (np.asarray(m.basic.area).ravel(), np.array(ma.area)),
            (np.asarray(m.basic.volume).ravel(), np.array(ma.volume)),
            (np.asarray(m.basic.mixing_length).ravel(), np.array(ma.mixing_length)),
            (np.asarray(m.basic_pressure).ravel(), np.array(ma.P_basic)),
            (np.asarray(m.staggered_pressure).ravel(), np.array(ma.P_stag)),
            (np.asarray(m._d_dr_transform), np.array(ma.d_dr_matrix)),
            (np.asarray(m._quantity_transform), np.array(ma.quantity_matrix)),
        ]
        for a_np, a_jax in checks:
            denom = np.maximum(np.abs(a_np), 1e-30)
            d = float(np.max(np.abs(a_np - a_jax) / denom))
            if d > max_rel_diff:
                max_rel_diff = d

    return max_rel_diff


def main() -> None:
    data_dir = Path('/Users/timlichtenberg/work/ssc-verify-task6/data/B')
    data_dir.mkdir(parents=True, exist_ok=True)
    out_csv = data_dir / 'B5_parity.csv'

    results = []

    # 1. BC 6
    diff_bc6 = verify_bc6_surface_flux(20, seed=42)
    results.append(('bc6_surface_flux', 20, diff_bc6, diff_bc6 < 1e-9))
    print(f'bc6_surface_flux: max_rel_diff = {diff_bc6:.3e}, pass = {diff_bc6 < 1e-9}')

    # 2. CMB law
    diff_cmb = verify_cmb_law_flux(20, seed=42)
    results.append(('cmb_law_flux', 20, diff_cmb, diff_cmb < 1e-9))
    print(f'cmb_law_flux: max_rel_diff = {diff_cmb:.3e}, pass = {diff_cmb < 1e-9}')

    # 3. MLT slope factor
    diff_mlt = verify_mlt_slope_factor(20, seed=42)
    results.append(('mlt_slope_factor', 20, diff_mlt, diff_mlt < 1e-9))
    print(f'mlt_slope_factor: max_rel_diff = {diff_mlt:.3e}, pass = {diff_mlt < 1e-9}')

    # 4. Mesh arrays
    diff_mesh = verify_mesh_arrays(20, seed=42)
    results.append(('mesh_arrays', 20, diff_mesh, diff_mesh < 1e-9))
    print(f'mesh_arrays: max_rel_diff = {diff_mesh:.3e}, pass = {diff_mesh < 1e-9}')

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['quantity', 'n_states', 'max_rel_diff', 'pass'])
        for q, n, d, p in results:
            writer.writerow([q, n, f'{d:.6e}', p])

    print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()
