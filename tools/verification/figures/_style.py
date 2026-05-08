"""Shared matplotlib style for Aragog V&V figures.

Single source of truth for colours, font sizes, and panel labelling so
the figure set looks consistent in the docs.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Tableau-10 inspired qualitative palette, hand-picked for print contrast.
PALETTE = {
    'numpy':      '#1f77b4',  # steel blue
    'jax':        '#d62728',  # brick red
    'analytic':   '#2ca02c',  # forest green
    'cond':       '#1f77b4',
    'conv':       '#ff7f0e',
    'grav':       '#2ca02c',
    'mix':        '#9467bd',
    'dil':        '#8c564b',
    'radio':      '#e377c2',
    'tidal':      '#7f7f7f',
    'total':      '#000000',
    # Permeability regimes
    'bkc':        '#1f77b4',
    'rg':         '#ff7f0e',
    'stokes':     '#2ca02c',
    # Radio isotopes
    'K40':        '#1f77b4',
    'U235':       '#ff7f0e',
    'U238':       '#2ca02c',
    'Th232':      '#9467bd',
    'Al26':       '#d62728',  # brick red — short-lived, dominant at t=0
    'Fe60':       '#8c564b',  # brown    — short-lived
}


def apply_rc():
    """Apply project-wide rc settings."""
    mpl.rcParams.update({
        'font.family':     'serif',
        'font.size':       10,
        'axes.labelsize':  10,
        'axes.titlesize':  10,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth':  0.8,
        'lines.linewidth': 1.4,
        'pdf.fonttype':    42,    # editable text in PDF
        'ps.fonttype':     42,
        'savefig.bbox':    'tight',
        'savefig.dpi':     300,
        'figure.dpi':      120,
    })


def panel_label(ax, text, loc='upper left', fontsize=10):
    """Add a bold (a)/(b)/(c) panel label to an axes."""
    coords = {
        'upper left':  (0.025, 0.95, 'left', 'top'),
        'upper right': (0.975, 0.95, 'right', 'top'),
        'lower left':  (0.025, 0.05, 'left', 'bottom'),
        'lower right': (0.975, 0.05, 'right', 'bottom'),
    }
    x, y, ha, va = coords[loc]
    ax.text(
        x, y, text, transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=fontsize, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none',
                  alpha=0.85, pad=1.5),
    )


def save(fig, path):
    """Save fig as both PDF and PNG with consistent settings.

    The docs site embeds the PNG (smaller, faster), and the PDF is
    used for high-quality reproduction in papers and reports. Both
    are written next to each other so the docs and paper paths share
    a single canonical figure.
    """
    from pathlib import Path
    p = Path(path)
    fig.savefig(p.with_suffix('.pdf'), format='pdf')
    fig.savefig(p.with_suffix('.png'), format='png')
    plt.close(fig)
