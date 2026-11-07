"""Schematic of the core alloy liquidus for ``docs/Explanations/core_alloy_liquidus.md``.

Temperature against pressure from 0 to 364 GPa (the centre of the Earth): the melting curve of
pure iron, the liquidus of iron with 2 and 4.8 wt% S and the Fe-S eutectic, all computed from
:class:`aragog.core.AlloyLiquidus` when the figure is built, and an example core adiabat from
the core-mantle boundary, 4500 K at 136 GPa. Numbered callouts sit in free space inside the
frame, chosen by a search that keeps them off every curve; their text and the legend lie beside
the frame. :func:`collisions` lists every text that touches a curve, another text or the frame.

The figure is written in the light and the dark theme of ``proteus_mpl`` to
``docs/figures/alloy_liquidus_sketch_{light,dark}.svg``.

Run:
    pip install 'fwl-aragog[verification]'
    python tools/verification/figures/sketch_alloy_liquidus.py
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from aragog.core import AlloyLiquidus, IronMeltingCurve, eutectic  # noqa: E402

OUT = Path(__file__).resolve().parents[3] / 'docs' / 'figures'
P = np.r_[
    np.arange(0.0, 40.0, 1.0), np.arange(40.0, 364.1, 4.0)
]  # GPa: fine at the low-P blend
P_CMB, T_CMB, SLOPE, CURV = 136.0, 4500.0, 6.7, 1.28866e-3  # example adiabat: K, K/GPa, K/GPa^2
DATA_LOW, DATA_HIGH = 15.0, 330.0  # GPa: non-ideal measured liquidus below, no data above
RMS_LOW, RMS_HIGH = 79, 153  # K, misfit below 15 GPa (docs/Validation/core_alloy_liquidus.md)
FONT = 10  # pt, every text of the figure
# cycle colours that both themes define: ink, red, orange, green, blue, purple
INK, CALLOUT, S2, EUTECTIC, ADIABAT, S48 = (f'C{i}' for i in range(6))


def adiabat(p):
    """Example core adiabat [K] at pressure ``p`` [GPa]."""
    x = np.asarray(p) - P_CMB
    return T_CMB + SLOPE * x + CURV * x * x


def curves() -> dict:
    """Temperatures [K] on ``P`` of pure iron, 2 and 4.8 wt% S, the Fe-S eutectic and the adiabat."""
    pa = P * 1e9
    out = {
        'Fe': np.asarray(IronMeltingCurve.t_melt_pure(pa), dtype=float),
        'S2': np.asarray(AlloyLiquidus({'S': 0.02}).t_melt(pa), dtype=float),
        'S48': np.asarray(AlloyLiquidus({'S': 0.048}).t_melt(pa), dtype=float),
        'TeS': np.asarray(eutectic('S', pa)[0], dtype=float),
    }
    out['adiabat'] = adiabat(P)
    return out


def icb_pressure() -> float:
    """Pressure [GPa] where the example adiabat falls below the melting curve of pure iron."""
    t_fe = IronMeltingCurve.t_melt_pure
    return brentq(lambda p: adiabat(p) - float(t_fe(p * 1e9)), P_CMB, P[-1], xtol=1e-6)


def notes(c: dict, p_icb: float) -> list[str]:
    """Text of the numbered callouts, with the numbers computed from the curves."""
    drop = float(np.interp(P_CMB, P, c['Fe']) - np.interp(P_CMB, P, c['S2']))
    return [
        f'Light elements lower the melting point: 2 wt% S melts {drop:.0f} K below pure iron at '
        'the core-mantle boundary (the 2 dots).',
        'Iron freezes where the core temperature falls below the liquidus. In this example pure '
        f'iron forms an inner core from {p_icb:.0f} GPa (shaded). With 2 wt% S the core stays '
        'liquid.',
        'Every Fe-S liquidus lies in the shaded band between pure iron and the eutectic.',
        f'Above {DATA_HIGH:.0f} GPa there are no data. The code extrapolates.',
        f'Below {DATA_LOW:.0f} GPa the measured Fe-S liquidus is not ideal. The curve misses it '
        f'by {RMS_LOW} to {RMS_HIGH} K (RMS).',
    ]


def _segments(ax):
    """Display-space vertices of every line of ``ax``, sampled at 20 points per segment."""
    pts = []
    for line in ax.get_lines():
        xy = line.get_transform().transform(np.column_stack(line.get_data()))
        for a, b in zip(xy[:-1], xy[1:]):
            pts.append(a + np.linspace(0, 1, 20)[:, None] * (b - a))
    return np.vstack(pts) if pts else np.empty((0, 2))


def _hits(bb, pts, pad=2.0) -> bool:
    """Whether a display box, grown by ``pad`` pixels, contains a point of ``pts``."""
    inside = (pts[:, 0] >= bb.x0 - pad) & (pts[:, 0] <= bb.x1 + pad)
    return bool(np.any(inside & (pts[:, 1] >= bb.y0 - pad) & (pts[:, 1] <= bb.y1 + pad)))


def _extent(t, rend):
    """Display box of a text with the patch drawn around it."""
    bb, patch = t.get_window_extent(rend), t.get_bbox_patch()
    if patch is None:
        return bb
    t.update_bbox_position_size(rend)  # an undrawn patch has no position yet
    return Bbox.union([bb, patch.get_window_extent(rend)])


def collisions(fig, ax) -> list[str]:
    """Texts of ``fig`` that touch a line of ``ax``, another text, or, inside ``ax``, its frame.

    Tick labels and the axis labels lie outside the frame and are left out; every other visible
    text of the figure is checked, the callouts inside the frame and the legend and notes beside
    it, each with the patch drawn around it.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts, frame = _segments(ax), ax.get_window_extent(rend)
    skip = set(ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label])
    texts = [
        t
        for t in fig.findobj(matplotlib.text.Text)
        if t.get_text() and t.get_visible() and t not in skip
    ]
    boxes = [(t, _extent(t, rend)) for t in texts]
    found = []
    for i, (t, bb) in enumerate(boxes):
        if bb.overlaps(frame):
            if _hits(bb, pts):
                found.append(f'{t.get_text()!r} touches a curve')
            inside = frame.x0 + 2 < bb.x0 and bb.x1 < frame.x1 - 2
            if not (inside and frame.y0 + 2 < bb.y0 and bb.y1 < frame.y1 - 2):
                found.append(f'{t.get_text()!r} touches the frame')
        for u, bu in boxes[i + 1 :]:
            if bb.overlaps(bu):
                found.append(f'{t.get_text()!r} touches {u.get_text()!r}')
    return found


def _callout(where, x, y, n, transform):
    """A numbered circle at (x, y) of ``where`` (an axes or a figure)."""
    box = dict(boxstyle='circle,pad=0.3', fc=plt.rcParams['axes.facecolor'], ec=CALLOUT, lw=1.0)
    return where.text(
        x,
        y,
        str(n),
        ha='center',
        va='center',
        fontsize=FONT,
        color=CALLOUT,
        bbox=box,
        transform=transform,
        zorder=6,
    )


def _place(fig, ax, n, candidates, taken):
    """Put callout ``n`` at the first candidate (data coordinates) clear of lines and texts."""
    rend = fig.canvas.get_renderer()
    frame, pts = ax.get_window_extent(rend), _segments(ax)
    for x, y in candidates:
        t = _callout(ax, x, y, n, ax.transData)
        bb = _extent(t, rend)
        ok = frame.x0 + 4 < bb.x0 and bb.x1 < frame.x1 - 4
        ok = ok and frame.y0 + 4 < bb.y0 and bb.y1 < frame.y1 - 4 and not _hits(bb, pts, 4.0)
        if ok and not any(bb.overlaps(q) for q in taken):
            taken.append(bb)
            return t
        t.remove()
    raise RuntimeError(f'no free spot for callout {n}')


def _around(p, t, radii=(16, 22, 28, 34, 40, 48, 56, 64)):
    """Candidate points [GPa, K] on rings around (p, t), in the scale of the frame."""
    out = []
    for r in radii:
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            out.append((p + r * 0.62 * np.cos(a), t + r * 17.6 * np.sin(a)))
    return out


def _wrap(text: str, width: int) -> str:
    """Wrap ``text``, keeping a number with its unit on one line."""
    text = text.replace(' wt% ', ' wt% ').replace(' K ', ' K ')
    return '\n'.join(textwrap.wrap(text, width))


def figure(theme: str):
    """The sketch in ``theme``: (figure, axes of the plot)."""
    import proteus_mpl

    proteus_mpl.use(theme)
    sizes = ('font.size', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize')
    plt.rcParams.update(dict.fromkeys(sizes + ('legend.fontsize',), FONT))
    c, p_icb = curves(), icb_pressure()
    t_icb = float(adiabat(p_icb))
    ink = plt.rcParams['text.color']
    fig = plt.figure(figsize=(10.4, 4.6))
    ax = fig.add_axes((0.075, 0.13, 0.53, 0.76))
    ax.axvspan(0, DATA_LOW, color=ink, alpha=0.07, lw=0)
    ax.axvspan(DATA_HIGH, P[-1], color=ink, alpha=0.07, lw=0)
    ax.fill_between(P, c['TeS'], c['Fe'], color=S2, alpha=0.14, lw=0)
    ic = P >= p_icb
    ax.fill_between(P[ic], c['adiabat'][ic], c['Fe'][ic], color=ADIABAT, alpha=0.30, lw=0)
    ax.plot(P, c['Fe'], color=INK, lw=1.8, label='pure iron')
    ax.plot(P, c['S2'], color=S2, lw=1.6, label='iron + 2 wt% S')
    ax.plot(P, c['S48'], color=S48, lw=1.6, label='iron + 4.8 wt% S')
    ax.plot(P, c['TeS'], color=EUTECTIC, lw=1.4, ls=(0, (6, 4)), label='Fe-S eutectic')
    up = P >= P_CMB
    ax.plot(P[up], c['adiabat'][up], color=ADIABAT, lw=1.6, ls=(0, (1.5, 3)),
            label='example core temperature profile (adiabat)')  # fmt: skip
    ax.plot([P_CMB, P_CMB], [1000, 7000], color=ink, lw=0.8, ls=(0, (3, 3)), alpha=0.7)
    t_fe, t_s2 = (float(np.interp(P_CMB, P, c[k])) for k in ('Fe', 'S2'))
    ax.plot([P_CMB, P_CMB], [t_fe, t_s2], color=CALLOUT, lw=1.4, marker='o', ms=5, zorder=5)
    ax.plot([p_icb], [t_icb], color=CALLOUT, marker='o', ms=6, zorder=5)
    ax.set(
        xlim=(0, P[-1]), ylim=(1000, 7000), xlabel='Pressure [GPa]', ylabel='Temperature [K]'
    )
    ax.grid(False)
    for x, text, ha in (
        (P_CMB, 'Earth: core-mantle boundary', 'center'),
        (P[-1], 'centre', 'right'),
    ):
        ax.text(x, 1.025, text, transform=ax.get_xaxis_transform(), ha=ha, va='bottom',
                fontsize=FONT, color=ink, alpha=0.8)  # fmt: skip
    fig.canvas.draw()
    taken = []

    def mid(p):
        return p, (np.interp(p, P, c['S2']) + np.interp(p, P, c['S48'])) / 2

    _place(fig, ax, 1, _around(P_CMB, (t_fe + t_s2) / 2), taken)
    _place(fig, ax, 2, _around(p_icb, t_icb), taken)
    _place(fig, ax, 3, _around(*mid(250)) + _around(*mid(200)), taken)
    _place(fig, ax, 4, [(347.0, t) for t in range(1400, 6800, 100)], taken)
    _place(fig, ax, 5, [(p, t) for t in range(6600, 1300, -100) for p in (7, 9, 11)], taken)
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.0), frameon=False,
                    handlelength=2.4, borderaxespad=0.0)  # fmt: skip
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(to_fig)
    y = bb.y0 - 0.05
    for n, text in enumerate(notes(c, p_icb), start=1):
        t = _callout(fig, bb.x0 + 0.008, y - 0.012, n, fig.transFigure)
        body = fig.text(bb.x0 + 0.03, y, _wrap(text, 46), ha='left', va='top', fontsize=FONT,
                        color=ink, linespacing=1.35)  # fmt: skip
        fig.canvas.draw()
        low = min(_extent(t, fig.canvas.get_renderer()).y0, body.get_window_extent().y0)
        y = to_fig.transform((0, low))[1] - 0.03
    return fig, ax


def _round_paths(svg: str) -> str:
    """Path coordinates of ``svg`` to 0.1 px: invisible, and it keeps the file below 40 kB."""
    num = re.compile(r'(-?\d+\.\d)\d+')
    return re.sub(r' d="[^"]*"', lambda m: num.sub(r'\1', m.group(0)), svg)


def main() -> None:
    """Write the sketch in both themes and refuse a figure with a collision."""
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in ('light', 'dark'):
        fig, ax = figure(theme)
        found = collisions(fig, ax)
        if found:
            raise RuntimeError(f'{theme}: ' + '; '.join(found))
        path = OUT / f'alloy_liquidus_sketch_{theme}.svg'
        with plt.rc_context({'svg.fonttype': 'none'}):  # text as text keeps the file small
            fig.savefig(path, metadata={'Date': None})
        path.write_text(_round_paths(path.read_text()))
        plt.close(fig)
        print('written', path, path.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
