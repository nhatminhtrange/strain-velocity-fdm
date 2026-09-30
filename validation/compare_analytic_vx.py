"""
Horizontal-velocity companion to compare_analytic.py (fine preset, dx = 1 m).

The amplitude factor is fitted on v_z, exactly as in the main comparison, and
then applied unchanged to v_x. Fitting v_x on its own would hide an error in
the horizontal-to-vertical amplitude ratio, which is precisely what the free
surface condition controls.

Writes results/analytic_comparison_fine_vx.{png,pdf}.

Usage:
    python compare_analytic_vx.py

Author: Minh Nhat Tran
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compare_analytic import (GreenfunctionFreesurface, OUT_DIR, PRESETS, RHO,
                              VP, VS, analytic_velocity, fit_scale, run_fd)

DX = 1.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = PRESETS['fine']
    offsets = cfg['offsets']
    mu = RHO * VS ** 2
    green = GreenfunctionFreesurface(RHO, RHO * VP ** 2 - 2 * mu, mu)
    cr = green.cd / green.zetar

    t_a, ana_z = analytic_velocity(green, cfg, component=1)
    _, ana_x = analytic_velocity(green, cfg, component=0)
    vz, t = run_fd(DX, cfg, component=1)
    vx, _ = run_fd(DX, cfg, component=0)

    scale = fit_scale(vz, t, t_a, ana_z, offsets)
    scale_x = fit_scale(vx, t, t_a, ana_x, offsets)
    print(f"scale fitted on vz: {scale:.4e}   on vx alone: {scale_x:.4e}   "
          f"ratio {scale_x / scale:.4f}")
    print(f"  {'offset':>7s} {'corr vx':>9s} {'peak ratio vx':>14s}")
    for j, o in enumerate(offsets):
        ref = np.interp(t, t_a, ana_x[o])
        c = np.corrcoef(vx[:, j] * scale, ref)[0, 1]
        r = np.abs(vx[:, j] * scale).max() / np.abs(ref).max()
        print(f"  {o:5.0f} m {c:9.4f} {r:14.4f}")

    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })
    t0 = 1.5 / cfg['fc']
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6))
    fig.subplots_adjust(hspace=0.35, wspace=0.22, left=0.10, right=0.97,
                        top=0.95, bottom=0.16)
    for i, (ax, o) in enumerate(zip(axs.flat, offsets)):
        ax.plot(t_a - t0, ana_x[o], color='k', lw=1.6, zorder=1)
        ax.plot(t - t0, vx[:, i] * scale, color='tab:red', lw=1.0,
                ls=(0, (4, 2)), zorder=2)
        ax.axvline(o / cr, color='0.5', lw=0.7, ls=':', zorder=0)
        ax.set_xlim(0, cfg['tmax'] - t0)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(7)
        ax.text(0.975, 0.90, f'{o:g} m', transform=ax.transAxes,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.22', fc='w', ec='0.6', lw=0.5))
        if i % 2 == 0:
            ax.set_ylabel('$v_x$')
        if i >= 2:
            ax.set_xlabel('Time (s)')

    handles = [Line2D([], [], color='k', lw=1.6, label='Analytic (Lamb)'),
               Line2D([], [], color='tab:red', lw=1.0, ls=(0, (4, 2)),
                      label=f'SV-FDM, dx = {DX:g} m'),
               Line2D([], [], color='0.5', lw=0.7, ls=':', label='$x/C_R$')]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.0), frameon=False)

    png = os.path.join(OUT_DIR, 'analytic_comparison_fine_vx.png')
    fig.savefig(png, dpi=600, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"saved {png}")


if __name__ == '__main__':
    main()
