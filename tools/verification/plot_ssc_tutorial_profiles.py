"""Write the radial profile figure of the solid-state convection tutorial.

Runs the Python blocks of ``docs/Tutorials/solid_state_convection.md`` (rheology off, the
default stagnant-lid closure and the yield case) on the tables in ``ARAGOG_TEST_EOS_DIR`` and
writes ``docs/assets/tutorials/ssc_tutorial_profiles.svg`` and its ``_dark`` variant: the
viscosities and the convective heat flux at the basic nodes against radius.

Usage: ``ARAGOG_TEST_EOS_DIR=<tables> python tools/verification/plot_ssc_tutorial_profiles.py``
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / 'docs' / 'Tutorials' / 'solid_state_convection.md'
OUT = ROOT / 'docs' / 'assets' / 'tutorials'
# Colour-blind safe (Tol bright), readable on white and on dark grey.
COLOURS = ('#4477AA', '#EE6677', '#228833', '#CCBB44')


def run_page() -> dict:
    """Namespace after the Python blocks of the tutorial page."""
    blocks = re.findall(r'```python\n(.*?)```', PAGE.read_text(encoding='utf-8'), re.DOTALL)
    ns: dict = {'__name__': '__main__'}
    for block in blocks:
        exec(compile(block, str(PAGE), 'exec'), ns)  # noqa: S102
    return ns


def profiles(ns: dict) -> dict:
    """Basic-node radius [km], log10 viscosities and convective flux [kW/m^2] of the 3 cases."""

    def basic(solver, name):
        return np.asarray(getattr(solver.state.phase_basic, name)()).ravel()

    off, on, yld = ns['solver_off'], ns['solver_on'], ns['solver_yield']
    return {
        'r_km': np.asarray(on.state._radii_basic).ravel() * 1e-3,
        'eta_off': np.log10(basic(off, 'eta_diff')),
        'eta_diff': np.log10(basic(on, 'eta_diff')),
        'eta_eff': np.log10(np.asarray(on.state.eta_eff_basic).ravel()),
        'eta_eff_yield': np.log10(np.asarray(yld.state.eta_eff_basic).ravel()),
        'r_lid_km': float(on.state.lid_state['r_lid_base']) * 1e-3,
        'q_off': np.asarray(off.state._jconv).ravel() * 1e-3,
        'q_on': np.asarray(on.state._jconv).ravel() * 1e-3,
        'q_yield': np.asarray(yld.state._jconv).ravel() * 1e-3,
    }


def draw(p: dict, dark: bool, path: Path) -> None:
    """One figure with the viscosity and the convective flux panels."""
    style = 'dark_background' if dark else 'default'
    with plt.style.context(style), plt.rc_context({'svg.fonttype': 'none'}):
        # text as text keeps each file below the 60 kB the page test allows
        fig, (ax_v, ax_q) = plt.subplots(1, 2, figsize=(6.8, 3.2), constrained_layout=True)
        if dark:
            fig.patch.set_facecolor('#1e1e1e')
            for ax in (ax_v, ax_q):
                ax.set_facecolor('#1e1e1e')
        r = p['r_km']
        c0, c1, c2, c3 = COLOURS
        ax_v.plot(r, p['eta_off'], color=c0, lw=1.4, label='Constant solid eta (off)')
        ax_v.plot(r, p['eta_eff'], color=c2, lw=1.4, label='Effective eta_eff (default)')
        ax_v.plot(r, p['eta_diff'], color=c1, lw=1.4, ls='--', label='Arrhenius eta_diff')
        ax_v.plot(
            r,
            p['eta_eff_yield'],
            color=c3,
            lw=1.4,
            ls=':',
            label='Effective eta_eff (c=150 Pa)',
        )
        ax_v.axvline(p['r_lid_km'], color='0.5', lw=1.0, ls='-.', label='Lid base r_lid')
        ax_v.set_xlabel('Radius [km]')
        ax_v.set_ylabel('log10(Viscosity [Pa s])')
        ax_v.legend(fontsize=6.5, loc='upper left', frameon=False)
        ax_q.plot(r, p['q_off'], color=c0, lw=1.4, label='Rheology off')
        ax_q.plot(r, p['q_on'], color=c2, lw=1.4, label='Rheology on (default)')
        ax_q.plot(r, p['q_yield'], color=c3, lw=1.4, ls=':', label='Yield active (c=150 Pa)')
        ax_q.set_xlabel('Radius [km]')
        ax_q.set_ylabel('Convective heat flux [kW/m^2]')
        ax_q.legend(fontsize=6.5, loc='upper left', frameon=False)
        fig.savefig(path, format='svg', facecolor=fig.get_facecolor())
        plt.close(fig)


def main() -> int:
    """Run the tutorial and write both figures; print the file paths."""
    p = profiles(run_page())
    OUT.mkdir(parents=True, exist_ok=True)
    for dark, name in (
        (False, 'ssc_tutorial_profiles.svg'),
        (True, 'ssc_tutorial_profiles_dark.svg'),
    ):
        draw(p, dark, OUT / name)
        print(OUT / name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
