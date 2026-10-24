"""Build and verify the PALEOS-mapped 175 kyr reference fixture ``checkpoint_profile_175kyr.npz``.

Provenance of ``tests/reference/checkpoint_profile_175kyr.npz`` (100 basic nodes,
index ``k = 0`` at the CMB ``r = 3480 km`` to ``k = 99`` at the surface ``r = 6371 km``):

- **Unchanged 175 kyr run arrays (all 100 nodes):** ``r_basic`` [m], ``P_basic`` [Pa],
  ``F_conv`` [W/m^2], ``F_tot`` [W/m^2], and ``v_unyielded`` [m/s] come directly from the
  175 kyr magma-ocean solidification snapshot on ``main``
  (commit ``056b1af8cf6963d1f9cd9c55618f2a20e84e3b24``).
- **Unchanged 175 kyr run nodes in ``T_basic`` and ``phi_basic``:**
  - At the 2 solid surface nodes ``k = 98, 99`` (where ``phi_basic == 0.0``), ``T_basic[98:100]``
    and ``phi_basic[98:100]`` are the unchanged 175 kyr run values (``1423.00 K`` and ``452.56 K``).
  - At the 94 interior mushy nodes ``k = 4..97``, ``phi_basic[4:98]`` are the unchanged 175 kyr
    run melt fractions.
- **Mapped arrays and nodes (98 of 100 ``T_basic`` nodes, 4 of 100 ``phi_basic`` nodes):**
  - The PALEOS ``MgSiO3`` liquidus :math:`T_\\mathrm{liq}(P)` and solidus
    :math:`T_\\mathrm{sol}(P) = 0.8\\,T_\\mathrm{liq}(P)` are evaluated at ``P_basic``.
  - At the 94 mushy nodes ``k = 4..97``, ``T_basic[4:98]`` is mapped from the run melt fraction
    ``phi_basic[4:98]`` by the PALEOS linear lever rule
    ``T_basic = T_sol + phi_basic * (T_liq - T_sol)``.
  - At the 4 deepest CMB boundary-layer nodes ``k = 0..3`` (where ``phi_old`` decreased toward
    the core in the Wolf & Bower 2018 run, which would otherwise invert ``dT/dr`` against the
    steeper PALEOS slope between ``k = 3`` and ``k = 4``), ``T_basic[0:4]`` preserves the 175 kyr
    run temperature offsets above node 4 (``T_new[0:4] = T_new[4] + (T_old[0:4] - T_old[4])``)
    so ``T_basic`` decreases strictly with radius across all 100 nodes, and ``phi_basic[0:4]`` is
    set to the PALEOS lever rule ``(T_new[0:4] - T_sol[0:4]) / (T_liq[0:4] - T_sol[0:4])``
    (matching the lever rule to ``4e-16``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'tools' / 'verification'))

from compute_solidus_reference_values import (  # noqa: E402
    PALEOS_MUSHY_ZONE_FACTOR,
    paleos_liquidus_temperature,
)

FIXTURE_PATH = Path(__file__).resolve().parent / 'checkpoint_profile_175kyr.npz'
MAIN_BASE_COMMIT = '056b1af8cf6963d1f9cd9c55618f2a20e84e3b24'

# Exact IEEE-754 float64 hex values of T_basic[0:5] from the 175 kyr run fixture at MAIN_BASE_COMMIT,
# used both to verify the git source when present and to allow offline/shallow-clone reconstruction.
_MAIN_T_BASIC_0_TO_4_HEX = (
    '0x1.2bad925fbbf40p+12',
    '0x1.2a84cea84cb9dp+12',
    '0x1.2974c8953454dp+12',
    '0x1.282d7665980e2p+12',
    '0x1.276c7ad014ab8p+12',
)

# SHA-256 digests of the unmapped 175 kyr run arrays and slices at MAIN_BASE_COMMIT, ensuring
# offline/shallow-clone verification catches any modification of unmapped run values.
_MAIN_UNMAPPED_SHA256 = {
    'r_basic': '3b6fd53a961ffdda27520f00a4983ee84bcdc926bc91c86a45e78df385b9e740',
    'P_basic': '1dc56144853593e76ab1c4f2368edfc0f17157b8dc31558d3340c0bb8082ed88',
    'F_conv': 'a31468470d7922b9cf6bb3a12fead53567b5bad7ab017e23dd9feb07f01c8b1c',
    'F_tot': 'd0bc7f542f78790ed069d5c0a222e1e6fc7eec5ab5f0fd51621dc6c991be8645',
    'v_unyielded': 'e3f764b066a1f675e29b773e611db54c4e025933479f1321267ccd301f9ef912',
    'T_basic[98:100]': 'e9ce03d58eaa6bff6607b0e57f986ae61fc80fbf04aa1c768d75ee2b26ae865e',
    'phi_basic[4:100]': '793f07452cc86c16f6dbeec1aaf2e53b53886a7de8985fc17075cea548061d43',
}


def load_source_arrays(
    source_npz: Path | None = None, source_commit: str = MAIN_BASE_COMMIT
) -> dict[str, np.ndarray]:
    """Load the 175 kyr source arrays from ``source_npz``, ``git show``, or the committed file."""
    if source_npz is not None:
        return dict(np.load(source_npz))
    try:
        raw = subprocess.check_output(
            [
                'git',
                '-C',
                str(REPO_ROOT),
                'show',
                f'{source_commit}:tests/reference/checkpoint_profile_175kyr.npz',
            ],
            stderr=subprocess.DEVNULL,
        )
        return dict(np.load(io.BytesIO(raw)))
    except Exception:
        d = dict(np.load(FIXTURE_PATH))
        t_old = d['T_basic'].copy()
        t_old[:5] = np.array(
            [float.fromhex(h) for h in _MAIN_T_BASIC_0_TO_4_HEX], dtype=np.float64
        )
        d['T_basic'] = t_old
        return d


def build_fixture_arrays(
    source_npz: Path | None = None, source_commit: str = MAIN_BASE_COMMIT
) -> dict[str, np.ndarray]:
    """Construct the PALEOS-mapped 175 kyr fixture dictionary from the 175 kyr run arrays."""
    d_old = load_source_arrays(source_npz=source_npz, source_commit=source_commit)
    expected_t0_4 = np.array(
        [float.fromhex(h) for h in _MAIN_T_BASIC_0_TO_4_HEX], dtype=np.float64
    )
    if not np.array_equal(d_old['T_basic'][:5], expected_t0_4):
        raise ValueError(
            'Source T_basic[0:5] does not match the 175 kyr run basal temperatures.'
        )
    unmapped_slices = {
        'r_basic': d_old['r_basic'],
        'P_basic': d_old['P_basic'],
        'F_conv': d_old['F_conv'],
        'F_tot': d_old['F_tot'],
        'v_unyielded': d_old['v_unyielded'],
        'T_basic[98:100]': d_old['T_basic'][98:100],
        'phi_basic[4:100]': d_old['phi_basic'][4:100],
    }
    for slice_name, slice_arr in unmapped_slices.items():
        digest = hashlib.sha256(
            np.ascontiguousarray(slice_arr, dtype=np.float64).tobytes()
        ).hexdigest()
        if digest != _MAIN_UNMAPPED_SHA256[slice_name]:
            raise ValueError(
                f'Unmapped run slice {slice_name!r} digest mismatch against 175 kyr snapshot.'
            )

    p_basic = d_old['P_basic']
    t_old = d_old['T_basic']
    phi_old = d_old['phi_basic'].copy()

    t_liq = paleos_liquidus_temperature(p_basic)
    t_sol = PALEOS_MUSHY_ZONE_FACTOR * t_liq

    t_new = np.where(phi_old > 0.0, t_sol + phi_old * (t_liq - t_sol), t_old)
    t_new[:4] = t_new[4] + (t_old[:4] - t_old[4])
    phi_new = phi_old.copy()
    phi_new[:4] = (t_new[:4] - t_sol[:4]) / (t_liq[:4] - t_sol[:4])

    if not np.all(np.diff(t_new) < 0.0):
        raise ValueError('Mapped T_basic must be strictly decreasing from CMB to surface.')
    if not np.array_equal(phi_new > 0.0, t_new > t_sol):
        raise ValueError('Mapped fixture violates phi_basic > 0 iff T_basic > T_sol(P_basic).')

    d_new = dict(d_old)
    d_new['T_basic'] = t_new
    d_new['phi_basic'] = phi_new
    return d_new


def serialize_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize ``arrays`` to ``.npz`` bytes in memory using ``np.savez``."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    """Build or check ``tests/reference/checkpoint_profile_175kyr.npz``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-npz',
        type=Path,
        default=None,
        help='Optional path to the unmapped 175 kyr Wolf & Bower (2018) .npz fixture.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=FIXTURE_PATH,
        help='Output .npz path (defaults to tests/reference/checkpoint_profile_175kyr.npz).',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Verify the committed .npz file matches the rebuilt arrays and bytes bitwise.',
    )
    args = parser.parse_args(argv)

    rebuilt = build_fixture_arrays(source_npz=args.source_npz)
    rebuilt_bytes = serialize_npz_bytes(rebuilt)

    if not args.check:
        args.output.write_bytes(rebuilt_bytes)

    committed = dict(np.load(args.output))
    if list(committed.keys()) != list(rebuilt.keys()):
        raise AssertionError(
            f'Key mismatch: committed {list(committed.keys())} vs rebuilt {list(rebuilt.keys())}'
        )
    for key, arr in rebuilt.items():
        if not np.array_equal(committed[key], arr):
            raise AssertionError(
                f'Array {key!r} in {args.output} is not bitwise equal to rebuilt.'
            )
    if args.output.read_bytes() != rebuilt_bytes:
        raise AssertionError(f'File bytes of {args.output} differ from rebuilt .npz archive.')

    print(
        f'OK: {args.output} bitwise verified (100 nodes; run arrays r_basic, P_basic, F_conv, '
        'F_tot, v_unyielded; mapped T_basic[0:98] and phi_basic[0:4]).'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
