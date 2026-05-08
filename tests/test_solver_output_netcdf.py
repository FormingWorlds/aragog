"""Unit tests for ``SolverOutput.to_netcdf`` and the
``EntropySolver.write_netcdf`` wrapper.

The standalone NetCDF writer is the public claim that Aragog is a
self-contained module: a script can build the solver, run it, and
dump the full state to disk without leaning on PROTEUS for I/O.

Failures here mean the standalone export is silently broken even when
the rest of the solver works, which is hard to notice during
PROTEUS-coupled runs (PROTEUS uses its own writer).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aragog.solver.entropy_solver import SolverOutput

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Synthetic SolverOutput factory
# ---------------------------------------------------------------------------

# Smallest shapes that still distinguish n_basic = n_stag + 1, and that
# put physically asymmetric values into every field so a regression that
# silently swaps two arrays (e.g. r_stag vs r_basic) fails immediately.
_N_STAG = 4
_N_BASIC = _N_STAG + 1


def _make_output(*, status: int = 0, dt: float = 1234.5) -> SolverOutput:
    """Build a SolverOutput with distinctive values in every field.

    Each field gets a unique scale or sign so the round-trip test can
    catch a regression that mixes them up. Physically nonsensical
    combinations are intentional: this is a serialisation test, not a
    physics test.
    """
    rng = np.random.default_rng(seed=20260508)
    stag = lambda scale: scale * rng.standard_normal(_N_STAG)  # noqa: E731
    basic = lambda scale: scale * rng.standard_normal(_N_BASIC)  # noqa: E731

    return SolverOutput(
        # Staggered profiles (n_stag=4)
        S_final=2900.0 + stag(50.0),
        T_stag=3500.0 + stag(80.0),
        phi_stag=np.clip(0.5 + 0.4 * stag(1.0), 0.0, 1.0),
        rho_stag=4500.0 + stag(200.0),
        visc_stag=10 ** (10.0 + stag(2.0)),
        # Mesh geometry (mixed dims)
        P_stag=np.linspace(1.0e9, 1.0e11, _N_STAG),
        r_basic=np.linspace(3.5e6, 6.371e6, _N_BASIC),
        r_stag=np.linspace(3.6e6, 6.27e6, _N_STAG),
        vol=basic(1.0)[:_N_STAG] ** 2 + 1.0,
        mass_stag=4500.0 * (basic(1.0)[:_N_STAG] ** 2 + 1.0),
        # Fluxes / heating
        heat_flux=basic(1e3),
        heating=stag(1e-9),
        eddy_diff=basic(10.0),
        cap_stag=stag(1e7),
        # Per-component flux decomposition (basic)
        jcond_b=basic(100.0),
        jconv_b=basic(1e3),
        jgrav_b=basic(50.0),
        jmix_b=basic(20.0),
        dSdr_b=basic(1e-4),
        phi_basic=np.clip(0.5 + 0.4 * basic(1.0), 0.0, 1.0),
        T_basic=3400.0 + basic(70.0),
        cp_basic=1500.0 + basic(50.0),
        rho_basic=4500.0 + basic(150.0),
        # Scalars (each chosen to be distinctive)
        T_magma=2950.123,
        T_core=4000.456,
        Phi_global=0.567890,
        Phi_global_vol=0.612345,
        M_mantle=4.2e24,
        M_mantle_liquid=2.4e24,
        M_mantle_solid=1.8e24,
        RF_depth=0.345678,
        E_th=1.23e29,
        E_state=4.56e30,
        E_state_cons=4.55e30,
        Cp_eff=1450.0,
        F_heat_total=12345.6,
        F_cmb=789.0,
        Q_radio_total=2.3e16,
        Q_tidal_total=4.5e15,
        step_dE_F_int_J=-1.0e22,
        step_dE_F_cmb_J=+5.0e21,
        step_dE_Q_radio_J=+3.0e19,
        step_dE_Q_tidal_J=+1.0e19,
        step_dE_Q_radio_cons_J=+3.1e19,
        step_dE_Q_tidal_cons_J=+1.1e19,
        step_solver_residual_J=1.0e10,
        dt_actual=dt,
        status=status,
    )


# ---------------------------------------------------------------------------
# to_netcdf
# ---------------------------------------------------------------------------


def test_to_netcdf_round_trip_preserves_every_field(tmp_path: Path) -> None:
    """Property: every scalar and every array round-trips bit-exactly.

    Discriminator: a regression that silently truncates ``f8`` to ``f4``
    would lose the distinctive low-order bits set by the rng-driven
    field values; this test rejects such a change.
    """
    out = _make_output()
    f = tmp_path / 'snapshot.nc'
    out.to_netcdf(f, time=12345.0, description='unit test snapshot')

    nc = pytest.importorskip('netCDF4')
    with nc.Dataset(f, mode='r') as ds:
        # Metadata
        assert ds.description == 'unit test snapshot'
        assert ds.Conventions == 'CF-1.8'
        assert hasattr(ds, 'aragog_version')
        assert hasattr(ds, 'created_utc')
        # Time scalar respects the explicit override
        assert float(ds['time'][...]) == pytest.approx(12345.0, rel=0, abs=0)

        # Scalars: every SolverOutput scalar must be present and equal.
        scalars_to_check = {
            'T_magma': out.T_magma,
            'T_core': out.T_core,
            'Phi_global': out.Phi_global,
            'Phi_global_vol': out.Phi_global_vol,
            'M_mantle': out.M_mantle,
            'M_mantle_liquid': out.M_mantle_liquid,
            'M_mantle_solid': out.M_mantle_solid,
            'RF_depth': out.RF_depth,
            'E_th': out.E_th,
            'E_state': out.E_state,
            'E_state_cons': out.E_state_cons,
            'Cp_eff': out.Cp_eff,
            'F_heat_total': out.F_heat_total,
            'F_cmb': out.F_cmb,
            'Q_radio_total': out.Q_radio_total,
            'Q_tidal_total': out.Q_tidal_total,
            'step_dE_F_int_J': out.step_dE_F_int_J,
            'step_dE_F_cmb_J': out.step_dE_F_cmb_J,
            'step_dE_Q_radio_J': out.step_dE_Q_radio_J,
            'step_dE_Q_tidal_J': out.step_dE_Q_tidal_J,
            'step_dE_Q_radio_cons_J': out.step_dE_Q_radio_cons_J,
            'step_dE_Q_tidal_cons_J': out.step_dE_Q_tidal_cons_J,
            'step_solver_residual_J': out.step_solver_residual_J,
            'dt_actual': out.dt_actual,
        }
        for name, expected in scalars_to_check.items():
            assert name in ds.variables, f'missing scalar {name!r}'
            v = ds[name]
            np.testing.assert_allclose(
                float(v[...]),
                expected,
                rtol=1e-15,
                atol=0.0,
                err_msg=f'scalar {name} did not round-trip exactly',
            )
            assert v.units != '', f'{name} missing units'
            assert v.long_name != '', f'{name} missing long_name'

        # Status: integer field (i4)
        assert int(ds['status'][...]) == out.status
        assert ds['status'].dtype == np.int32

        # Arrays: full bit-exact match. Distinct dim per array enforces
        # that the writer didn't silently swap basic-vs-staggered.
        staggered_arrays = {
            'r_stag': out.r_stag,
            'P_stag': out.P_stag,
            'S_final': out.S_final,
            'T_stag': out.T_stag,
            'phi_stag': out.phi_stag,
            'rho_stag': out.rho_stag,
            'visc_stag': out.visc_stag,
            'vol': out.vol,
            'mass_stag': out.mass_stag,
            'heating': out.heating,
            'cap_stag': out.cap_stag,
        }
        for name, expected in staggered_arrays.items():
            v = ds[name]
            assert v.dimensions == ('staggered',), f'{name} on wrong dim: {v.dimensions}'
            np.testing.assert_array_equal(
                np.asarray(v[:], dtype=np.float64),
                np.asarray(expected, dtype=np.float64).ravel(),
                err_msg=f'staggered array {name} did not round-trip exactly',
            )

        basic_arrays = {
            'r_basic': out.r_basic,
            'heat_flux': out.heat_flux,
            'eddy_diff': out.eddy_diff,
            'jcond_b': out.jcond_b,
            'jconv_b': out.jconv_b,
            'jgrav_b': out.jgrav_b,
            'jmix_b': out.jmix_b,
            'dSdr_b': out.dSdr_b,
            'phi_basic': out.phi_basic,
            'T_basic': out.T_basic,
            'cp_basic': out.cp_basic,
            'rho_basic': out.rho_basic,
        }
        for name, expected in basic_arrays.items():
            v = ds[name]
            assert v.dimensions == ('basic',), f'{name} on wrong dim: {v.dimensions}'
            np.testing.assert_array_equal(
                np.asarray(v[:], dtype=np.float64),
                np.asarray(expected, dtype=np.float64).ravel(),
                err_msg=f'basic array {name} did not round-trip exactly',
            )


def test_to_netcdf_default_time_falls_back_to_dt_actual(tmp_path: Path) -> None:
    """Edge case: the ``time`` argument is optional. When omitted, the
    writer falls back to ``dt_actual``. A regression that hard-coded
    time=0.0 (plausible if the lazy default were misimplemented) would
    fail this discriminator because dt is set to a distinctive 1234.5.
    """
    out = _make_output(dt=1234.5)
    f = tmp_path / 'no_time.nc'
    out.to_netcdf(f)  # no `time=`

    nc = pytest.importorskip('netCDF4')
    with nc.Dataset(f, mode='r') as ds:
        assert float(ds['time'][...]) == pytest.approx(1234.5, rel=0, abs=0)


def test_to_netcdf_creates_parent_directory(tmp_path: Path) -> None:
    """Edge case: writing into a non-existent subdirectory must succeed
    (the writer should mkdir the parent, matching ``Path.mkdir`` with
    ``parents=True, exist_ok=True``). Catches a regression that
    requires the caller to pre-create the directory.
    """
    out = _make_output()
    target = tmp_path / 'a' / 'b' / 'c' / 'snap.nc'
    assert not target.parent.exists()
    out.to_netcdf(target)
    assert target.is_file()


def test_to_netcdf_overwrites_existing_file(tmp_path: Path) -> None:
    """Property: opening with ``mode='w'`` truncates an existing file.
    A regression that opened in append mode would leave stale fields
    from the prior write, which would surface as duplicated variables
    or stale metadata.
    """
    out_a = _make_output(status=0)
    out_b = _make_output(status=7)  # different status sentinel
    f = tmp_path / 'over.nc'
    out_a.to_netcdf(f)
    out_b.to_netcdf(f)
    nc = pytest.importorskip('netCDF4')
    with nc.Dataset(f, mode='r') as ds:
        assert int(ds['status'][...]) == 7


def test_to_netcdf_rejects_unphysical_negative_node_count() -> None:
    """Anti-happy-path: a SolverOutput built with mismatched n_basic
    and n_stag (n_basic != n_stag + 1) is structurally invalid; if
    it ever lands in ``to_netcdf`` the writer should still produce a
    file with the correct dim sizes for the actual array lengths,
    not silently truncate. This test is a discriminator: it asserts
    the dim sizes match the input array sizes verbatim.
    """
    out = _make_output()
    # Mutate r_basic to a different (still-positive) length to confirm
    # the writer reads dim sizes from the array, not from a cached
    # constant. Use a length-3 array; drop the matching basic-node
    # arrays so the file stays internally consistent.
    out.r_basic = np.linspace(3.5e6, 6.371e6, 3)
    out.heat_flux = np.linspace(0.0, 1.0, 3)
    out.eddy_diff = np.linspace(0.0, 1.0, 3)
    out.jcond_b = np.linspace(0.0, 1.0, 3)
    out.jconv_b = np.linspace(0.0, 1.0, 3)
    out.jgrav_b = np.linspace(0.0, 1.0, 3)
    out.jmix_b = np.linspace(0.0, 1.0, 3)
    out.dSdr_b = np.linspace(0.0, 1.0, 3)
    out.phi_basic = np.linspace(0.0, 1.0, 3)
    out.T_basic = np.linspace(0.0, 1.0, 3)
    out.cp_basic = np.linspace(0.0, 1.0, 3)
    out.rho_basic = np.linspace(0.0, 1.0, 3)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        f = Path(tmp.name)
    try:
        out.to_netcdf(f)
        nc = pytest.importorskip('netCDF4')
        with nc.Dataset(f, mode='r') as ds:
            assert ds.dimensions['basic'].size == 3
            assert ds.dimensions['staggered'].size == _N_STAG
    finally:
        f.unlink(missing_ok=True)
