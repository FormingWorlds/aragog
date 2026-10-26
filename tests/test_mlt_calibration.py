"""Calibrated viscous mixing length of the solid-state MLT closure.

Module under test: ``aragog.rheology.viscous_mixing_length_factor`` and
``aragog.rheology.compute_mlt_top_slope``. With rheology enabled, the viscous
branch uses ``l_v = min(s_bot (r - r_in), s_top(theta) (r_out - r))`` in place
of the Abe length ``l``: the viscous eddy diffusivity scales by ``(l_v / l)^4``
and the inviscid branch keeps ``l``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aragog.rheology import (
    SolidRheologyParams,
    compute_mlt_top_slope,
    viscous_mixing_length_factor,
)
from tools.verification._ssc_helpers import (
    korenaga_ti_fixed_point as _korenaga_ti_fixed_point,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

TOP = -3  # Node near the surface where the top slope sets l_v.
BOTTOM = 3  # Node near the CMB where the bottom slope sets l_v.
RE_CRIT = 9.0 / 8.0


def _mlt_kappa_h(
    log10visc: float,
    radii: np.ndarray,
    mixing_length: np.ndarray,
    q_factor: np.ndarray | float = 1.0,
    xp=np,
):
    """Evaluate the local MLT viscous-inviscid eddy diffusivity blend with factor q."""
    rho = 4000.0
    cp = 1200.0
    alpha = 3.0e-5
    g = 9.81
    temp = 2000.0
    drive = 1.0e-6
    nu = (10.0**log10visc) / rho

    prefactor = alpha * g * temp * drive / cp
    v_visc_abe = prefactor * (mixing_length**3) / (18.0 * nu)
    v_invisc = xp.sqrt(prefactor * (mixing_length**2) / 16.0)

    v_visc = q_factor * v_visc_abe
    reynolds = q_factor * (v_visc * mixing_length / nu)
    blend_width = 0.01 * RE_CRIT
    w_invisc = 0.5 * (1.0 + xp.tanh((reynolds - RE_CRIT) / blend_width))
    v_conv = (1.0 - w_invisc) * v_visc + w_invisc * v_invisc
    return v_conv * mixing_length


@pytest.mark.physics_invariant
def test_factor_is_the_fourth_power_of_the_length_ratio():
    """Top slope 0.5 gives 0.5^4 near the surface, 1 near the CMB, and 1 at boundaries."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    q = viscous_mixing_length_factor(r, r[0], r[-1], ml, 0.5, 1.0)
    assert q[-2] == pytest.approx(0.0625, rel=1e-12)
    assert q[1] == pytest.approx(1.0, rel=1e-12)
    assert q[0] == pytest.approx(1.0, rel=0.0) and q[-1] == pytest.approx(1.0, rel=0.0)
    np.testing.assert_array_equal(
        viscous_mixing_length_factor(r, r[0], r[-1], ml, 1.0, 1.0), 1.0
    )


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_theta_dependent_top_slope_hand_values_and_limits():
    """Verify s_top(theta) = s_top_inf + (1 - s_top_inf) exp(-c_theta max(theta, 0))."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    s_inf = 0.22
    c_th = 0.15

    # 1. Isoviscous limit (theta -> 0): s_top(0) = 1.0 exactly, recovering Abe length.
    assert compute_mlt_top_slope(0.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        1.0, rel=1e-15
    )
    q_zero = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=0.0, c_theta=c_th
    )
    np.testing.assert_allclose(q_zero, 1.0, rtol=1e-15)

    # 2. Negative theta (inverted temperature difference): max(theta, 0) = 0 clamps s_top at 1.0.
    assert compute_mlt_top_slope(-5.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        1.0, rel=1e-15
    )
    q_neg = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=-5.0, c_theta=c_th
    )
    np.testing.assert_allclose(q_neg, 1.0, rtol=1e-15)

    # 3. Hand-computed intermediate value at theta = 15:
    # s_top(15) = 0.22 + 0.78 * exp(-0.15 * 15) = 0.22 + 0.78 * exp(-2.25) = 0.3022113951582542.
    expected_s15 = 0.22 + 0.78 * math.exp(-2.25)
    expected_q15 = expected_s15**4
    assert compute_mlt_top_slope(15.0) == pytest.approx(expected_s15, rel=1e-14)
    assert compute_mlt_top_slope(15.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        expected_s15, rel=1e-14
    )
    q_15 = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=15.0, c_theta=c_th
    )
    assert q_15[TOP] == pytest.approx(expected_q15, rel=1e-14)
    assert q_15[BOTTOM] == pytest.approx(1.0, rel=1e-14)

    # 4. Stagnant-lid asymptotic limit (theta >> 8, e.g. theta = 250): s_top -> s_top_inf = 0.22.
    assert compute_mlt_top_slope(250.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        s_inf, rel=1e-13
    )
    q_inf = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=250.0, c_theta=c_th
    )
    assert q_inf[TOP] == pytest.approx(s_inf**4, rel=1e-13)
    assert q_inf[BOTTOM] == pytest.approx(1.0, rel=1e-14)


@pytest.mark.physics_invariant
def test_viscous_mixing_length_factor_theta_dispatch_and_limits(monkeypatch):
    """Verify viscous_mixing_length_factor evaluates compute_mlt_top_slope when theta is given."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    s_inf = 0.22
    c_th = 0.15

    # In the isoviscous limit (theta = 0), s_top(0) = 1.0, recovering the Abe mixing length factor.
    q_zero = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=0.0, c_theta=c_th
    )
    assert q_zero[TOP] == pytest.approx(1.0, rel=1e-14)

    # When theta is None, s_top defaults to s_top_inf, yielding s_top_inf^4 at the surface.
    q_none = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=None
    )
    assert q_none[TOP] == pytest.approx(s_inf**4, rel=1e-14)

    # At finite theta = 15.0, factor matches analytical compute_mlt_top_slope^4.
    expected_s15 = compute_mlt_top_slope(15.0, s_top_inf=s_inf, c_theta=c_th)
    q_15 = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=15.0, c_theta=c_th
    )
    assert q_15[TOP] == pytest.approx(expected_s15**4, rel=1e-14)

    # Monkeypatching compute_mlt_top_slope proves viscous_mixing_length_factor delegates to it.
    monkeypatch.setattr('aragog.rheology.compute_mlt_top_slope', lambda *a, **kw: 0.5)
    q_patched = viscous_mixing_length_factor(r, r[0], r[-1], ml, theta=15.0)
    assert q_patched[TOP] == pytest.approx(0.5**4, rel=1e-14)


@pytest.mark.physics_invariant
def test_viscous_mixing_length_factor_bottom_slope_calibration():
    """Verify viscous_mixing_length_factor with non-unit bottom_slope scales the CMB boundary layer."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    # At the CMB boundary node (node 0), mixing_length is zero; factor is 1.0.
    # At node 1 (near-CMB), radii - r_in governs l_v; with bottom_slope=0.5, l_v / l = 0.5.
    q_05 = viscous_mixing_length_factor(r, r[0], r[-1], ml, bottom_slope=0.5)
    assert q_05[0] == 1.0
    assert q_05[1] == pytest.approx(0.5**4, rel=1e-12)
    assert q_05[BOTTOM] == pytest.approx(0.5**4, rel=1e-12)


@pytest.mark.physics_invariant
def test_branch_switch_stays_continuous_in_kappa():
    """Sweep viscosity over 20 decades at the near-surface node for constant and theta slopes.

    Scaling the switch Reynolds number by q keeps the transition at kappa_v = kappa_i,
    so ln(kappa_h) never steps by more than the viscous slope (ln 10 * 0.02 = 0.046).
    """
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    lvs = np.arange(-2.0, 18.0, 0.02)

    cases = [
        (viscous_mixing_length_factor(r, r[0], r[-1], ml, s_top_inf=0.5), 0.5**4),
        (viscous_mixing_length_factor(r, r[0], r[-1], ml, s_top_inf=0.22), 0.22**4),
        (
            viscous_mixing_length_factor(
                r, r[0], r[-1], ml, s_top_inf=0.22, theta=15.0, c_theta=0.15
            ),
            (0.22 + 0.78 * math.exp(-2.25)) ** 4,
        ),
    ]
    for q_vec, q_expected in cases:
        k = np.array([_mlt_kappa_h(lv, r, ml, q_vec)[TOP] for lv in lvs])
        step = np.abs(np.diff(np.log(k)))
        assert step.max() < 0.05
        assert np.all(k > 0.0)
        k_ref = [_mlt_kappa_h(lv, r, ml, 1.0)[TOP] for lv in (lvs[0], lvs[-1])]
        assert k[0] / k_ref[0] == pytest.approx(1.0, rel=1e-10)
        assert k[-1] / k_ref[1] == pytest.approx(q_expected, rel=1e-10)


@pytest.mark.physics_invariant
def test_numpy_and_jax_theta_top_slope_parity():
    """Verify NumPy and JAX evaluate identical s_top(theta) and mixing-length factors."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    r_j = jnp.asarray(r)
    ml_j = jnp.asarray(ml)

    for th in (-3.0, 0.0, 8.0, 15.0, 40.0):
        s_np = compute_mlt_top_slope(th, s_top_inf=0.22, c_theta=0.15, xp=np)
        s_jx = float(
            compute_mlt_top_slope(jnp.asarray(th), s_top_inf=0.22, c_theta=0.15, xp=jnp)
        )
        assert s_jx == pytest.approx(s_np, rel=1e-14)

        q_np = viscous_mixing_length_factor(
            r, r[0], r[-1], ml, s_top_inf=0.22, bottom_slope=1.0, xp=np, theta=th, c_theta=0.15
        )
        q_jx = np.asarray(
            viscous_mixing_length_factor(
                r_j,
                r_j[0],
                r_j[-1],
                ml_j,
                s_top_inf=0.22,
                bottom_slope=1.0,
                xp=jnp,
                theta=jnp.asarray(th),
                c_theta=0.15,
            )
        )
        np.testing.assert_allclose(q_jx, q_np, rtol=1e-14, atol=1e-30)


def test_solid_rheology_params_top_slope_defaults_and_bounds():
    """Verify SolidRheologyParams defaults and bounds validation for s_top_inf and c_theta."""
    params = SolidRheologyParams()
    assert not hasattr(params, 'mlt_top_slope')
    assert params.mlt_bottom_slope == pytest.approx(1.0, rel=1e-15)
    assert params.s_top_inf == pytest.approx(0.22, rel=1e-15)
    assert params.c_theta == pytest.approx(0.15, rel=1e-15)

    try:
        custom = SolidRheologyParams(s_top_inf=0.18, c_theta=0.0)
    except ValueError:
        pytest.fail('c_theta = 0.0 must be accepted')
    assert custom.s_top_inf == pytest.approx(0.18, rel=1e-15)
    assert custom.c_theta == pytest.approx(0.0, abs=1e-15)

    with pytest.raises(ValueError, match='s_top_inf must be in'):
        SolidRheologyParams(s_top_inf=0.0)
    with pytest.raises(ValueError, match='s_top_inf must be in'):
        SolidRheologyParams(s_top_inf=-0.1)
    with pytest.raises(ValueError, match='s_top_inf must be in'):
        SolidRheologyParams(s_top_inf=1.1)
    with pytest.raises(ValueError, match='mlt_bottom_slope must be in'):
        SolidRheologyParams(mlt_bottom_slope=0.0)
    with pytest.raises(ValueError, match='mlt_bottom_slope must be in'):
        SolidRheologyParams(mlt_bottom_slope=1.1)
    with pytest.raises(ValueError, match='c_theta must be non-negative'):
        SolidRheologyParams(c_theta=-0.01)


def test_korenaga_ti_fixed_point_raises_on_non_convergence() -> None:
    """Verify _korenaga_ti_fixed_point raises RuntimeError when iteration fails to converge."""
    z_grid = np.linspace(0.01, 0.99, 50)
    t_prof = 273.0 + 1500.0 * z_grid * (1.0 - 0.5 * z_grid)
    with pytest.raises(RuntimeError, match='within 2 steps'):
        _korenaga_ti_fixed_point(t_prof, z_grid, 273.0, 1500.0, max_iter=2, tol_k=1.0e-15)
    with pytest.raises(RuntimeError, match='lid exceeds domain'):
        _korenaga_ti_fixed_point(t_prof, z_grid, 273.0, 100.0)


@pytest.mark.unit
def test_verification_benchmark_generators_edge_cases_and_schema(tmp_path) -> None:
    """Verify edge-case guards, failure reporting, and clean schema in benchmark generators."""
    import copy
    import json
    from pathlib import Path

    from tools.verification import (
        generate_ssc_0d_reference_benchmarks as ref0d_mod,
    )
    from tools.verification import (
        generate_ssc_scaling_exponents_benchmarks as scaling_mod,
    )
    from tools.verification import (
        generate_ssc_solver_wiring_benchmarks as wiring_mod,
    )

    wiring_path = Path(__file__).parent / 'reference' / 'ssc_solver_wiring_benchmarks.json'
    wiring_data = json.loads(wiring_path.read_text(encoding='utf-8'))

    # 1. Missing --eos-dir in generation mode must fail with exit code 1 without mutating file
    orig_wiring_text = wiring_path.read_text(encoding='utf-8')
    wiring_copy = tmp_path / 'wiring_copy.json'
    wiring_copy.write_text(orig_wiring_text, encoding='utf-8')
    rc_missing_eos = wiring_mod.main(
        ['--output', str(wiring_copy), '--eos-dir', str(tmp_path / 'does_not_exist')]
    )
    assert rc_missing_eos == 1
    assert wiring_copy.read_text(encoding='utf-8') == orig_wiring_text

    # 2. Incomplete EOS directory in verify_benchmarks appends failure instead of raising
    empty_eos_dir = tmp_path / 'empty_eos'
    empty_eos_dir.mkdir()
    missing_table_failures = wiring_mod.verify_benchmarks(wiring_data, eos_dir=empty_eos_dir)
    assert any('Missing required table file' in f for f in missing_table_failures)

    # 3. abe_solid_default parity bound, local call-count deltas, starts_block dict check, rationale
    assert wiring_data['rhs_parity_and_activity']['abe_solid_default']['rationale'].startswith(
        'Default state set on the CI SPIDER EOS tables: s_init=2900 J/kg/K'
    )
    wiring_bad_def = copy.deepcopy(wiring_data)
    wiring_bad_def['rhs_parity_and_activity']['abe_solid_default'][
        'rel_rhs_numpy_vs_jax_per_state'
    ][0] = 1.0e-3
    assert any(
        'abe_solid_default.max_rel_rhs_numpy_vs_jax' in f
        for f in wiring_mod.verify_benchmarks(wiring_bad_def, eos_dir=None)
    )

    wiring_bad_local = copy.deepcopy(wiring_data)
    wiring_bad_local['rhs_wall_time']['six_row_call_count_table'][0][
        'measured_numpy_local_delta'
    ] = 99
    assert any(
        'local' in f.lower()
        for f in wiring_mod.verify_benchmarks(wiring_bad_local, eos_dir=None)
    )

    wiring_bad_starts = copy.deepcopy(wiring_data)
    wiring_bad_starts['perturbed_starts']['abe_solid_default'] = None
    assert any(
        'abe_solid_default' in f
        for f in wiring_mod.verify_benchmarks(wiring_bad_starts, eos_dir=None)
    )

    # 4. compute_crossing_time_linear returns None (JSON null) when threshold is never crossed or already exceeded at t=0
    t_grid = np.array([0.0, 1.0e7, 2.0e7], dtype=float)
    err_grid = np.array([0.01, 0.02, 0.03], dtype=float)
    assert ref0d_mod.compute_crossing_time_linear(t_grid, err_grid, 0.22) is None
    assert (
        ref0d_mod.compute_crossing_time_linear(
            t_grid, np.array([0.30, 0.35, 0.40], dtype=float), 0.22
        )
        is None
    )
    assert ref0d_mod.compute_crossing_time_linear(t_grid, err_grid, 0.025) == pytest.approx(
        1.5e7, rel=1.0e-12
    )

    # 5. korenaga_ti_fixed_point domain safety and non-convergence RuntimeError
    with pytest.raises(RuntimeError, match='lid exceeds domain'):
        scaling_mod.korenaga_ti_fixed_point(
            np.array([2000.0, 2000.0]), np.array([0.1, 0.2]), 300.0, 1000.0
        )
    with pytest.raises(RuntimeError, match='within 1 steps'):
        scaling_mod.korenaga_ti_fixed_point(
            np.array([1500.0, 300.0]), np.array([0.1, 0.9]), 300.0, 2000.0, max_iter=1
        )

    # 6. Clean metadata and cases_spec in ssc_scaling_exponents_benchmarks.json and primary fit helper
    scaling_path = Path(__file__).parent / 'reference' / 'ssc_scaling_exponents_benchmarks.json'
    scaling_data = json.loads(scaling_path.read_text(encoding='utf-8'))
    assert tuple(scaling_data['metadata'].keys()) == scaling_mod.EXPECTED_META_KEYS
    assert all(
        tuple(spec.keys()) == scaling_mod.EXPECTED_SPEC_KEYS
        for spec in scaling_data['cases_spec']
    )
    prim_fit = scaling_mod.compute_primary_fit_all_12_nu_korenaga(scaling_data['cases_results'])
    assert prim_fit['multilinear']['alpha'] == pytest.approx(
        float(scaling_data['primary_fit_all_12_nu_korenaga']['multilinear']['alpha']),
        rel=1.0e-12,
    )
    assert prim_fit['multilinear']['beta'] == pytest.approx(
        float(scaling_data['primary_fit_all_12_nu_korenaga']['multilinear']['beta']),
        rel=1.0e-12,
    )


@pytest.mark.unit
def test_benchmark_generators_require_origin_commit_and_clean_tree(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify benchmark provenance format in --check and origin/clean-tree rules at generation time."""
    import copy
    import json
    import subprocess
    from pathlib import Path

    from tools.verification import (
        generate_ssc_0d_reference_benchmarks as ref0d_mod,
    )
    from tools.verification import (
        generate_ssc_scaling_exponents_benchmarks as scaling_mod,
    )
    from tools.verification import (
        generate_ssc_solver_wiring_benchmarks as wiring_mod,
    )
    from tools.verification._ssc_helpers import (
        _is_commit_on_origin,
        _is_valid_commit_sha,
        _resolve_origin_head_commit,
    )

    ref_dir = Path(__file__).parent / 'reference'
    wiring_data = json.loads((ref_dir / 'ssc_solver_wiring_benchmarks.json').read_text('utf-8'))
    ref0d_data = json.loads((ref_dir / 'ssc_0d_reference_benchmarks.json').read_text('utf-8'))
    scaling_data = json.loads(
        (ref_dir / 'ssc_scaling_exponents_benchmarks.json').read_text('utf-8')
    )

    # Set up a real git repository in tmp_path with one commit on origin/main and one local-only commit
    tmp_repo = tmp_path / 'git_repo'
    tmp_repo.mkdir()
    git_env = {
        'GIT_AUTHOR_NAME': 'Test',
        'GIT_AUTHOR_EMAIL': 'test@example.com',
        'GIT_COMMITTER_NAME': 'Test',
        'GIT_COMMITTER_EMAIL': 'test@example.com',
        'PATH': '/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin',
    }
    subprocess.run(['git', '-C', str(tmp_repo), 'init', '-b', 'main'], check=True, env=git_env)
    (tmp_repo / 'tracked.txt').write_text('v1\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(tmp_repo), 'add', 'tracked.txt'], check=True, env=git_env)
    subprocess.run(
        ['git', '-C', str(tmp_repo), 'commit', '-m', 'Initial commit'], check=True, env=git_env
    )
    origin_sha = subprocess.check_output(
        ['git', '-C', str(tmp_repo), 'rev-parse', 'HEAD'], text=True
    ).strip()
    subprocess.run(
        ['git', '-C', str(tmp_repo), 'update-ref', 'refs/remotes/origin/main', origin_sha],
        check=True,
        env=git_env,
    )
    assert _is_commit_on_origin(origin_sha, tmp_repo) is True
    assert _resolve_origin_head_commit(tmp_repo) == origin_sha

    # Create a real second commit on local branch 'main' that is NOT on refs/remotes/origin/*
    (tmp_repo / 'tracked.txt').write_text('v2\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(tmp_repo), 'add', 'tracked.txt'], check=True, env=git_env)
    subprocess.run(
        ['git', '-C', str(tmp_repo), 'commit', '-m', 'Local unpushed commit'],
        check=True,
        env=git_env,
    )
    local_only_sha = subprocess.check_output(
        ['git', '-C', str(tmp_repo), 'rev-parse', 'HEAD'], text=True
    ).strip()
    assert _is_valid_commit_sha(local_only_sha) is True
    assert _is_commit_on_origin(local_only_sha, tmp_repo) is False
    with pytest.raises(RuntimeError, match='origin'):
        _resolve_origin_head_commit(tmp_repo)

    # Reset back to origin_sha and dirty the tracked file to verify the dirty-tree guard
    subprocess.run(
        ['git', '-C', str(tmp_repo), 'reset', '--hard', origin_sha], check=True, env=git_env
    )
    (tmp_repo / 'tracked.txt').write_text('dirty\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='uncommitted'):
        _resolve_origin_head_commit(tmp_repo)

    assert _is_valid_commit_sha('uncommitted') is False
    assert _is_commit_on_origin('uncommitted', tmp_repo) is False

    specs = (
        ('wiring', wiring_mod, wiring_data, ('canary', 'aragog_commit')),
        ('ref0d', ref0d_mod, ref0d_data, ('metadata', 'git_commit')),
        ('scaling', scaling_mod, scaling_data, ('metadata', 'git_commit')),
    )
    for label, mod, data, (sec_key, sha_key) in specs:
        sha = str(data[sec_key][sha_key])
        assert _is_valid_commit_sha(sha) is True, f'{label} {sha} is not a 40-hex SHA'

        bad_data = copy.deepcopy(data)
        bad_data[sec_key][sha_key] = 'not-a-valid-40-hex-sha'
        failures = (
            mod.verify_benchmarks(bad_data, eos_dir=None)
            if label == 'wiring'
            else mod.verify_benchmarks(bad_data)
        )
        assert any('commit' in f or 'SHA' in f for f in failures), (
            f'{label} verify_benchmarks accepted malformed SHA'
        )

        out_copy = tmp_path / f'{label}_copy.json'
        orig_text = json.dumps(data, indent=2) + '\n'
        out_copy.write_text(orig_text, encoding='utf-8')
        with monkeypatch.context() as mp:
            mp.setattr(
                mod,
                '_resolve_origin_head_commit',
                lambda *_a, **_kw: (_ for _ in ()).throw(
                    RuntimeError('HEAD is not on origin or working tree is dirty')
                ),
            )
            argv = ['--output', str(out_copy)]
            if label == 'wiring':
                eos_dir = tmp_path / 'dummy_eos'
                eos_dir.mkdir(exist_ok=True)
                argv.extend(['--eos-dir', str(eos_dir)])
            assert mod.main(argv) == 1
            assert out_copy.read_text(encoding='utf-8') == orig_text


@pytest.mark.unit
def test_solid_state_convection_tutorial_registered_and_code_blocks_parse() -> None:
    """Verify the solid-state convection tutorial, its TOML/Python blocks, and its light/dark SVGs."""
    import ast
    import re
    import tomllib
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    mkdocs_text = (repo_root / 'mkdocs.yml').read_text(encoding='utf-8')
    assert 'Tutorials/solid_state_convection.md' in mkdocs_text

    tut_path = repo_root / 'docs' / 'Tutorials' / 'solid_state_convection.md'
    assert tut_path.is_file()
    tut_text = tut_path.read_text(encoding='utf-8')

    svg_light = repo_root / 'docs' / 'assets' / 'tutorials' / 'ssc_tutorial_profiles.svg'
    svg_dark = repo_root / 'docs' / 'assets' / 'tutorials' / 'ssc_tutorial_profiles_dark.svg'
    assert svg_light.is_file() and 1000 < svg_light.stat().st_size <= 60_000
    assert svg_dark.is_file() and 1000 < svg_dark.stat().st_size <= 60_000

    toml_blocks = re.findall(r'```toml\n(.*?)```', tut_text, flags=re.DOTALL)
    assert len(toml_blocks) >= 1
    abe_solid_text = (repo_root / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml').read_text(
        encoding='utf-8'
    )
    parsed_toml = tomllib.loads(abe_solid_text + '\n' + toml_blocks[0])
    assert parsed_toml['phase_solid']['rheology']['enabled'] is True
    assert parsed_toml['phase_solid']['rheology']['stress_closure_mode'] == 'lid'

    py_blocks = re.findall(r'```python\n(.*?)```', tut_text, flags=re.DOTALL)
    assert len(py_blocks) >= 3
    combined_source = '\n'.join(py_blocks)
    tree = ast.parse(combined_source)
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == 'run_case' for node in tree.body
    )


@pytest.mark.unit
def test_half_cell_surface_flux_patch_pins_bc5_prescribed_surface_temperature() -> None:
    """Pin the half_cell_surface_flux verification helper against unpatched BC 5 on case_06."""
    import json
    from pathlib import Path

    from aragog.solver.entropy_solver import EntropySolver
    from tools.verification._ssc_helpers import half_cell_surface_flux
    from tools.verification.generate_ssc_scaling_exponents_benchmarks import (
        build_scaling_case_params,
    )

    ref_path = Path(__file__).parent / 'reference' / 'ssc_scaling_exponents_benchmarks.json'
    data = json.loads(ref_path.read_text(encoding='utf-8'))
    spec = next(s for s in data['cases_spec'] if s['case_id'] == 'case_06')
    snap = data['steady_state_snapshots']['case_06']
    s_final = np.array(snap['S_final'], dtype=float)
    t_surf = float(spec['T_surf_K'])
    k_cond = float(spec.get('k_thermal_W_per_m_K', 4.0))

    with half_cell_surface_flux():
        solver_patched = EntropySolver(build_scaling_case_params(spec), entropy_eos=None)
        solver_patched.initialize()
        rhs_patched = np.asarray(solver_patched._dSdt_single(0.0, s_final), dtype=float)
        f_top_patched = float(solver_patched.state._heat_flux[-1])
        t_cell = float(
            np.asarray(solver_patched.state.phase_staggered.temperature()).ravel()[-1]
        )
        r_b = np.asarray(solver_patched.state._evaluator.mesh.basic.radii).ravel()
        dr_half = 0.5 * float(r_b[-1] - r_b[-2])

    expected_half_cell_flux = k_cond * (t_cell - t_surf) / dr_half
    assert f_top_patched == pytest.approx(expected_half_cell_flux, rel=1.0e-14)
    assert f_top_patched == pytest.approx(float(snap['F_top_W_m2']), rel=1.0e-12)
    assert float(np.max(np.abs(rhs_patched))) < 1.0e-9

    # Without half_cell_surface_flux, shipped EntropyState ignores outer_boundary_condition = 5:
    # outer_boundary_value = 273 K and 1000 K yield bit-identical RHS and lower top flux.
    params_273 = build_scaling_case_params(spec)
    params_273.boundary_conditions.outer_boundary_value = 273.0
    solver_273 = EntropySolver(params_273, entropy_eos=None)
    solver_273.initialize()
    rhs_273 = np.asarray(solver_273._dSdt_single(0.0, s_final), dtype=float)
    f_top_unpatched = float(solver_273.state._heat_flux[-1])

    params_1000 = build_scaling_case_params(spec)
    params_1000.boundary_conditions.outer_boundary_value = 1000.0
    solver_1000 = EntropySolver(params_1000, entropy_eos=None)
    solver_1000.initialize()
    rhs_1000 = np.asarray(solver_1000._dSdt_single(0.0, s_final), dtype=float)

    assert np.array_equal(rhs_273, rhs_1000)
    assert f_top_unpatched == pytest.approx(5.387415282426097, rel=1.0e-12)
    assert f_top_patched > f_top_unpatched * 1.5
    assert float(np.max(np.abs(rhs_273))) > 1.0e-3
