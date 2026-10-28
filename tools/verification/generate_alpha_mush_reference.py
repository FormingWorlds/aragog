"""Write the mush-state thermal expansivity of the NumPy phase evaluator to a CSV.

The states are melt fractions 0.1, 0.5 and 0.9 at 12 pressures from 0 to 145 GPa under
``cp_blend = 'latent'``. Each row holds alpha at ``matprop_smooth_width`` 0 (the two-phase
value alone, clamped) and at 0.01 (blended with the single-phase value). The header records
the commit of the aragog tree that ran and the sha256 of every table file read.

Usage::

    python tools/verification/generate_alpha_mush_reference.py <P-S table dir> <out.csv>
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

import aragog
from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator


def main(eos_dir: str, out: str) -> None:
    """Evaluate the mush states on the tables of ``eos_dir`` and write ``out``."""
    eos = EntropyEOS(eos_dir)
    P1 = np.linspace(0.0, 145e9, 12)
    S_sol, S_liq = np.ravel(eos.solidus_entropy(P1)), np.ravel(eos.liquidus_entropy(P1))
    phi = np.repeat([0.1, 0.5, 0.9], P1.size)
    P = np.tile(P1, 3)
    S = np.tile(S_sol, 3) + phi * np.tile(S_liq - S_sol, 3)
    alpha = {}
    for width in (0.0, 0.01):
        ev = EntropyPhaseEvaluator(eos, 10.0, matprop_smooth_width=width)
        ev.set_pressure(P)
        ev.set_entropy(S)
        ev.update()
        alpha[width] = np.ravel(ev.thermal_expansivity())
    root = Path(aragog.__file__).resolve().parents[2]
    commit = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    header = [f'aragog commit {commit}', f'tables {Path(eos_dir).name}']
    for f in sorted(Path(eos_dir).glob('*.dat')):
        header.append(f'sha256 {f.name} {hashlib.sha256(f.read_bytes()).hexdigest()}')
    header.append('P_Pa,S_J_kg_K,melt_fraction,alpha_width0_per_K,alpha_width0p01_per_K')
    np.savetxt(
        out,
        np.column_stack([P, S, phi, alpha[0.0], alpha[0.01]]),
        delimiter=',',
        fmt='%.17e',
        header='\n'.join(header),
    )


if __name__ == '__main__':
    main(*sys.argv[1:3])
