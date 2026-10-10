"""Schematic of the ``core_module`` core boundary condition.

Panel (a) shows the structure and the energy flows: the inner core, the outer
core on the adiabat, the optional resolved shell that holds the stable layer,
the CMB, the mantle boundary layer and the bottom mantle cell, with the terms
of the core energy budget as arrows. Panel (b) shows the temperature against
radius: the adiabat, the melting curve, their crossing at the inner-core
boundary, the optional stable layer and the temperature drop across the mantle
boundary layer. Neither panel is to scale; the curves are illustrative shapes,
not solver output.

The figure is written in the light and the dark theme of ``proteus_mpl`` to
``docs/figures/core_module_sketch_{light,dark}.png``, with a PDF beside each.

Run:
    pip install 'fwl-aragog[verification]'
    python tools/verification/figures/sketch_core_module.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import proteus_mpl  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Wedge  # noqa: E402

OUT = Path(__file__).resolve().parents[3] / 'docs' / 'figures'

# Radii of the sketch (not to scale): inner core, shell base, CMB, boundary layer, mantle cell.
R_ICB, R_SH, R_CMB, R_BL, R_TOP = 0.40, 0.80, 1.00, 1.07, 1.26
TH0, TH1 = 38.0, 142.0  # opening of the wedge [deg]


def palette(theme: str) -> dict:
    """Colours and fill opacities of one theme.

    Parameters
    ----------
    theme : str
        ``'light'`` or ``'dark'``.

    Returns
    -------
    dict
        Named colours, and the opacities of the core, the boundary layer and
        the mantle cell.
    """
    c = proteus_mpl.COLORS
    dark = theme == 'dark'
    core_a, bl_a, cell_a = (0.30, 0.55, 0.22) if dark else (0.45, 0.38, 0.12)
    return dict(
        dark=dark,
        ink=c['paper'] if dark else c['ink'],
        fog=c['fog'],
        mist=c['mist'],
        heat=c['magma'],
        # The mantle red of the light theme is too dark on the dark background.
        deep='#E08373' if dark else c['mantle'],
        gold=c['solar'] if dark else c['solar_deep'],
        blue=c['azure'] if dark else c['ocean'],
        core_a=core_a,
        bl_a=bl_a,
        cell_a=cell_a,
    )


def draw_structure(ax, col: dict) -> None:
    """Draw panel (a): a wedge of the core and the mantle base with the energy flows.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the panel.
    col : dict
        Colours from :func:`palette`.
    """
    ink, fog, heat, deep, gold = (col[k] for k in ('ink', 'fog', 'heat', 'deep', 'gold'))

    def band(r0, r1, color, alpha=1.0, **kw):
        ax.add_patch(
            Wedge((0, 0), r1, TH0, TH1, width=r1 - r0, facecolor=color, alpha=alpha, **kw)
        )

    def arc(r, **kw):
        th = np.radians(np.linspace(TH0, TH1, 100))
        ax.plot(r * np.cos(th), r * np.sin(th), **kw)

    def heat_arrow(r0, r1, deg, label, lx, ly, lw=2.2, ha='left'):
        th = np.radians(deg)
        p0, p1 = (r0 * np.cos(th), r0 * np.sin(th)), (r1 * np.cos(th), r1 * np.sin(th))
        style = dict(arrowstyle='-|>', mutation_scale=15, lw=lw, color=heat, zorder=5)
        ax.add_patch(FancyArrowPatch(p0, p1, **style))
        ax.text(lx, ly, label, color=heat, va='center', ha=ha, fontsize=11)

    def side_label(r, y, text, color):
        th = np.radians(TH0)
        ax.plot([r * np.cos(th) + 0.012, 1.08], [r * np.sin(th), y], color=fog, lw=0.7)
        # Kept out of the layout, so the length of a label does not rescale the panels.
        ax.text(1.10, y, text, color=color, va='center', fontsize=10).set_in_layout(False)

    band(0.0, R_ICB, fog, 0.85, edgecolor='none')
    band(R_ICB, R_CMB, col['mist'], col['core_a'], edgecolor='none')
    shell = dict(edgecolor=gold, linestyle='--', linewidth=1.2, hatch='...')
    band(R_SH, R_CMB, 'none', 1.0, **shell)
    band(R_CMB, R_BL, heat, col['bl_a'], edgecolor='none')
    band(R_BL, R_TOP, heat, col['cell_a'], edgecolor='none')
    arc(R_ICB, color=ink, lw=1.2)
    arc(R_CMB, color=ink, lw=2.2)
    arc(R_BL, color=heat, lw=0.9, ls=':')

    centred = dict(ha='center', va='center')
    ax.text(0.0, 0.24, 'inner core\n(solid)', color='white', fontsize=10, **centred)
    ax.text(0.0, 0.565, 'outer core', color=ink, fontsize=10.5, **centred)
    ax.text(0.0, 0.495, '(liquid, on the adiabat)', color=ink, fontsize=9, **centred)
    ax.text(-0.18, 1.165, 'bottom mantle cell', color=deep, fontsize=10, **centred)

    heat_arrow(R_ICB + 0.02, R_ICB + 0.16, 135, r'$Q_L$', -0.44, 0.335, ha='right')
    heat_arrow(R_ICB + 0.02, R_ICB + 0.16, 45, r'$Q_g$', 0.44, 0.335)
    heat_arrow(0.62, 0.76, 127, r'$Q_s$', -0.33, 0.665, ha='center')
    heat_arrow(0.62, 0.76, 53, r'$Q_R$', 0.33, 0.665, ha='center')
    heat_arrow(R_BL + 0.015, R_TOP + 0.10, 68, r'$Q_\mathrm{cmb}$', 0.56, 1.30, lw=4.2)

    side_label(R_TOP - 0.07, 0.90, r'$T_m$: bottom cell at the CMB pressure', deep)
    side_label(0.5 * (R_CMB + R_BL), 0.76, r'boundary layer, thickness $\delta$', heat)
    side_label(R_CMB, 0.62, r'CMB: $r_\mathrm{cmb}$, $T_\mathrm{cmb}$ (solver state)', ink)
    shell_text = r'shell with the stable layer (optional), base $r_\mathrm{sh}$'
    side_label(0.5 * (R_SH + R_CMB), 0.48, shell_text, gold)
    side_label(R_ICB, 0.30, r'ICB: $r_\mathrm{icb}(T_\mathrm{cmb})$', ink)
    balance = (
        r'$Q_\mathrm{cmb} = A_\mathrm{cmb}\,q = -\tilde{C}\,\dfrac{dT_\mathrm{cmb}}{dt} + Q_R$'
    )
    ax.text(1.10, 0.115, balance, color=ink, fontsize=11, va='center')
    ax.text(1.10, -0.06, r'$\tilde{C} = C_s + C_L + C_g$', color=ink, fontsize=11, va='center')
    ax.set_xlim(-1.02, 2.72)
    ax.set_ylim(-0.14, 1.42)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Structure and energy flows', loc='left', fontsize=12)


def draw_temperature(bx, col: dict) -> None:
    """Draw panel (b): the temperature against radius, as illustrative curves.

    Parameters
    ----------
    bx : matplotlib.axes.Axes
        Axes of the panel.
    col : dict
        Colours from :func:`palette`.
    """
    ink, fog, heat, deep, gold = (col[k] for k in ('ink', 'fog', 'heat', 'deep', 'gold'))
    r = np.linspace(0.0, R_CMB, 300)
    t_cmb, d2, t_m = 1.00, 3.4, 0.80
    adiabat = t_cmb * np.exp((R_CMB**2 - r**2) / d2)
    t_icb = t_cmb * np.exp((R_CMB**2 - R_ICB**2) / d2)
    # Steeper than the adiabat and equal to it at the ICB: above it inside, below it outside.
    melt = t_icb + 0.75 * (R_ICB**2 - r**2)

    bx.axvspan(0, R_ICB, color=fog, alpha=0.30 if col['dark'] else 0.18, lw=0)
    bx.axvspan(R_CMB, R_BL, color=heat, alpha=col['bl_a'] * 0.6, lw=0)
    bx.axvspan(R_BL, R_TOP, color=heat, alpha=col['cell_a'] * 0.7, lw=0)
    for x, c, ls in ((R_ICB, fog, '-'), (R_SH, gold, '--'), (R_CMB, ink, '-')):
        bx.axvline(x, color=c, lw=0.8, ls=ls, zorder=0)

    bx.plot(r, melt, color=col['blue'], lw=1.8)
    bx.plot(r[r >= R_ICB], adiabat[r >= R_ICB], color=ink, lw=2.2)
    bx.plot(r[r <= R_ICB], adiabat[r <= R_ICB], color=ink, lw=1.0, ls=':')
    bx.plot([R_ICB], [t_icb], 'o', color=ink, ms=6, zorder=6)
    bx.plot([R_CMB, R_BL], [t_cmb, t_m], color=heat, lw=2.2)
    bx.plot([R_BL, R_TOP], [t_m, t_m - 0.012], color=deep, lw=2.2)
    # The optional stable layer is warmer than the adiabat and meets it at the shell base.
    rs = np.linspace(R_SH, R_CMB, 50)
    excess = 0.06 * ((rs - R_SH) / (R_CMB - R_SH)) ** 2
    bx.plot(rs, np.interp(rs, r, adiabat) + excess, color=gold, lw=1.6, ls='--')

    thin = dict(arrowstyle='-', color=fog, lw=0.7)
    bx.text(0.435, 0.935, 'melting curve', color=col['blue'], fontsize=10, ha='left')
    bx.text(0.555, 1.265, r'adiabat $T_a(r)$', color=ink, fontsize=10)
    icb_text = 'inner-core boundary:\nadiabat = melting curve'
    bx.text(0.03, 1.10, icb_text, fontsize=9, color=ink, va='top')
    bx.annotate('', (R_ICB - 0.012, t_icb - 0.012), (0.24, 1.105), arrowprops=thin)
    layer_text = dict(color=gold, fontsize=9, ha='left', va='bottom')
    bx.text(R_SH + 0.012, 1.225, 'stable layer\n(optional)', **layer_text)
    span = dict(arrowstyle='<->', color=heat, lw=1.2)
    bx.annotate('', (R_TOP + 0.045, t_cmb), (R_TOP + 0.045, t_m), arrowprops=span)
    bx.plot([R_CMB + 0.005, R_TOP + 0.07], [t_cmb, t_cmb], color=heat, lw=0.7, ls=':')
    drop = dict(color=heat, fontsize=11, va='center')
    bx.text(R_TOP + 0.065, 0.5 * (t_cmb + t_m), r'$\Delta T$', **drop)
    bx.text(
        R_CMB - 0.02, t_cmb - 0.055, r'$T_\mathrm{cmb}$', color=ink, fontsize=10, ha='right'
    )
    bx.text(R_TOP, t_m - 0.065, r'$T_m$', color=deep, fontsize=10, ha='right')
    bx.text(0.5 * (R_CMB + R_BL), 1.375, r'$\delta$', color=heat, fontsize=11, ha='center')
    law = r'$q = \dfrac{k\,\Delta T}{\min(\delta,\ \Delta r_{1/2})}$'
    bx.text(0.03, 0.675, law, color=heat, fontsize=12, va='bottom')

    bx.set_xlim(0.0, R_TOP + 0.17)
    bx.set_ylim(0.63, 1.43)
    bx.set_xticks([0.0, R_ICB, R_SH, R_CMB])
    bx.set_xticklabels(['0', r'$r_\mathrm{icb}$', r'$r_\mathrm{sh}$', r'$r_\mathrm{cmb}$'])
    bx.set_yticks([])
    bx.set_xlabel('radius')
    bx.set_ylabel('temperature')
    bx.grid(False)
    bx.set_title('(b) Temperature against radius (not to scale)', loc='left', fontsize=12)


def main() -> None:
    """Write the sketch in both themes."""
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in ('light', 'dark'):
        proteus_mpl.use(theme)
        col = palette(theme)
        ratios = {'height_ratios': [1.0, 1.35]}
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.6, 8.6), gridspec_kw=ratios)
        draw_structure(ax, col)
        draw_temperature(bx, col)
        fig.tight_layout()
        stem = OUT / f'core_module_sketch_{theme}'
        fig.savefig(stem.with_suffix('.png'), dpi=170)
        fig.savefig(stem.with_suffix('.pdf'))
        plt.close(fig)
        print('written', stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
