"""
Validate the free-surface implementation against the analytic Lamb solution.

A vertical line force on the surface of a homogeneous half-space has a
closed-form response (2D Lamb problem). Comparing against it separates the
free-surface boundary condition from every other source of error: no model
complexity, no reference solver, and the Rayleigh arrival is exact.

The analytic solution is computed by lamb_2dhalf_surface (Ki-Tae Kim,
https://github.com/ktkimit/lamb_2dhalf_surface), which evaluates the Miklowitz
Green's function and convolves it with a Ricker wavelet, treating the Rayleigh
pole as a Cauchy principal value. It is LGPL-3.0 and therefore not vendored
here; this script fetches it into external/ on first run.

Both solutions are compared as particle velocity: the analytic displacement is
differentiated in time to match what the FD solver records.

Two presets, selected with PRESET:

    coarse   fc =  6 Hz, dx = 20 / 10 / 5 m   (3.5 to 14 points per wavelength)
    fine     fc = 20 Hz, dx =  4 /  2 / 1 m   (5.3 to 21 points per wavelength)

Writes results/analytic_comparison_<preset>.{png,pdf}.

Usage:
    python compare_analytic.py            # both presets
    PRESET=fine python compare_analytic.py

Author: Minh Nhat Tran
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CASE_DIR))
import jax.numpy as jnp
from fwi import forward_jax, ricker_jax

UPSTREAM = 'https://raw.githubusercontent.com/ktkimit/lamb_2dhalf_surface/master/lamb2d_freesurface.py'


def fetch_lamb_solver():
    """Download the LGPL-3.0 reference solver into external/ if not present.

    It is kept out of this repository so that the MIT licence here stays
    unencumbered; the file is used unmodified apart from dropping the
    demonstration block at its end, which would otherwise run on import.
    """
    ext = os.path.join(CASE_DIR, 'external')
    path = os.path.join(ext, 'lamb2d_freesurface.py')
    if not os.path.exists(path):
        from urllib.request import urlopen
        os.makedirs(ext, exist_ok=True)
        print(f'fetching {UPSTREAM}')
        text = urlopen(UPSTREAM, timeout=60).read().decode()
        marker = "# Young's modulus"
        if marker in text:
            text = text[:text.index(marker)].rstrip() + '\n'
        with open(path, 'w') as f:
            f.write(text)
        with open(os.path.join(ext, 'README.md'), 'w') as f:
            f.write('# Third-party code (not part of this repository)\n\n'
                    'lamb2d_freesurface.py is downloaded from\n'
                    'https://github.com/ktkimit/lamb_2dhalf_surface\n'
                    '(Ki-Tae Kim, LGPL-3.0). Only the demonstration block at\n'
                    'the end of the file is removed so it can be imported.\n')
    sys.path.insert(0, ext)


fetch_lamb_solver()
from lamb2d_freesurface import GreenfunctionFreesurface, Ricker

OUT_DIR = os.path.join(CASE_DIR, 'results')

VP, VS, RHO = 2000.0, 1155.0, 2000.0   # Poisson solid, Vp/Vs = sqrt(3)
FD_ORDER = 8
NT_ANALYTIC = 500                      # the analytic solution is slow per sample
FD_STABILITY = {4: 9 / 8 + 1 / 24,
                8: 1225 / 1024 + 245 / 3072 + 49 / 5120 + 5 / 7168}

PRESETS = {
    'coarse': dict(fc=6.0, tmax=2.5, lx=3000.0, lz=1200.0, src_x=500.0,
                   pml=600.0, offsets=[600.0, 1000.0, 1400.0, 1800.0],
                   grids=[20.0, 10.0, 5.0]),
    'fine':   dict(fc=20.0, tmax=0.75, lx=600.0, lz=250.0, src_x=100.0,
                   pml=120.0, offsets=[120.0, 220.0, 320.0, 420.0],
                   grids=[4.0, 2.0, 1.0]),
}
COLORS = ['tab:blue', 'tab:orange', 'tab:red']


def analytic_velocity(green, cfg):
    """Vertical particle velocity at each offset, on a shared time axis.

    The quadrature in the reference implementation occasionally fails to
    converge on the last few samples, leaving isolated spikes well after the
    wave has passed. They are replaced by the local median, which leaves the
    signal untouched and only removes the outliers.
    """
    ricker = Ricker(green, 1.0, cfg['fc'], 1.5 / cfg['fc'])
    t = np.linspace(0.0, cfg['tmax'], NT_ANALYTIC)
    out = {}
    for x in cfg['offsets']:
        v = np.gradient(
            np.array([ricker.integration_convolution(x, ti)[1] for ti in t]), t)
        # a sample is an outlier if it dwarfs both of its neighbours
        pad = np.r_[v[0], v, v[-1]]
        neighbour = np.maximum(np.abs(pad[:-2]), np.abs(pad[2:]))
        bad = np.abs(v) > 5 * np.maximum(neighbour, 1e-3 * np.abs(v).max())
        if bad.any():
            v[bad] = np.interp(t[bad], t[~bad], v[~bad])
        out[x] = v
    return t, out


def run_fd(dx, cfg, order=FD_ORDER):
    """Vertical velocity at the surface receivers on a dx grid."""
    nx = int(round(cfg['lx'] / dx)) + 1
    nz = int(round(cfg['lz'] / dx)) + 1
    vs = np.full((nz, nx), VS)
    vp = np.full((nz, nx), VP)
    rho = np.full((nz, nx), RHO)

    dt = 0.4 / (FD_STABILITY[order] * VP * np.sqrt(2) / dx)
    nt = int(cfg['tmax'] / dt)
    rec_ix = np.array([int(round((cfg['src_x'] + o) / dx))
                       for o in cfg['offsets']])

    wavelet = ricker_jax(jnp.arange(nt) * dt, cfg['fc'], 1.5 / cfg['fc'])
    vz = np.asarray(forward_jax(
        vs, vp, rho, int(round(cfg['src_x'] / dx)), 0,
        rec_ix, np.zeros(len(rec_ix), dtype=int),
        nx, nz, dx, dx, dt, nt, cfg['fc'], int(round(cfg['pml'] / dx)),
        wavelet, block_size=nt, fd_order=order, free_surface=True)[1])
    return vz, np.arange(nt) * dt


def fit_scale(vz, t, t_a, analytic, offsets):
    """One amplitude factor shared across offsets.

    The analytic solution and the FD solver use different source
    normalisations, but the offset dependence (geometrical spreading) must be
    reproduced, so the factor is fitted on all offsets simultaneously rather
    than per trace.
    """
    num = den = 0.0
    for j, o in enumerate(offsets):
        ref = np.interp(t, t_a, analytic[o])
        num += np.dot(vz[:, j], ref)
        den += np.dot(vz[:, j], vz[:, j])
    return num / den


def plot_preset(name, cfg, cr, t_a, analytic, fd, scales, path):
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })
    offsets, grids = cfg['offsets'], cfg['grids']
    t0 = 1.5 / cfg['fc']
    n_rows = (len(offsets) + 1) // 2
    colors = dict(zip(grids, COLORS))

    fig = plt.figure(figsize=(7.2, 2.4 + 2.0 * n_rows))
    gs = fig.add_gridspec(1 + n_rows, 2, height_ratios=[0.7] + [1.0] * n_rows,
                          hspace=0.5, wspace=0.22,
                          left=0.10, right=0.97, top=0.97, bottom=0.10)

    # (a) geometry
    km = 1000.0
    ax = fig.add_subplot(gs[0, :])
    ax.add_patch(plt.Rectangle((0, 0), cfg['lx'] / km, cfg['lz'] / km,
                               fc='0.92', ec='0.4', lw=0.8))
    ax.axhline(0, color='k', lw=1.6)
    ax.plot(cfg['src_x'] / km, 0, marker='*', ms=14, mfc='r', mec='k',
            mew=0.6, ls='none', clip_on=False, zorder=5)
    ax.plot([(cfg['src_x'] + o) / km for o in offsets], [0] * len(offsets),
            marker='v', ms=7, mfc='w', mec='k', mew=0.8, ls='none',
            clip_on=False, zorder=5)
    for o in offsets:
        ax.annotate(f'{o:g} m', ((cfg['src_x'] + o) / km, 0),
                    xytext=(0, -13), textcoords='offset points',
                    ha='center', fontsize=7)
    ax.text(cfg['lx'] / (2 * km), cfg['lz'] / (2 * km),
            f'$V_P$ = {VP:.0f} m/s\n$V_S$ = {VS:.0f} m/s\n'
            fr'$\rho$ = {RHO:.0f} kg/m$^3$' f'\n$C_R$ = {cr:.1f} m/s',
            ha='center', va='center', fontsize=8.5)
    ax.text(0.012, 0.90, 'free surface', transform=ax.transAxes, fontsize=7.5)
    ax.set(xlim=(0, cfg['lx'] / km), ylim=(cfg['lz'] / km, -0.12 * cfg['lz'] / km),
           xlabel='Distance (km)', ylabel='Depth (km)')
    ax.text(-0.075, 1.06, '(a)', transform=ax.transAxes, fontweight='bold')

    # (b) waveforms at their real amplitudes (one shared scale factor)
    for i, o in enumerate(offsets):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        ax.plot(t_a - t0, analytic[o], color='k', lw=1.6, zorder=1)
        for dx in grids:
            vz, t = fd[dx]
            ax.plot(t - t0, vz[:, i] * scales[dx], color=colors[dx],
                    lw=1.0, ls=(0, (4, 2)), zorder=2)
        ax.axvline(o / cr, color='0.5', lw=0.7, ls=':', zorder=0)
        ax.set_xlim(0, cfg['tmax'] - t0)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(7)
        ax.text(0.975, 0.90, f'{o:g} m', transform=ax.transAxes,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.22', fc='w', ec='0.6', lw=0.5))
        if i % 2 == 0:
            ax.set_ylabel('$v_z$')
        if i // 2 == n_rows - 1:
            ax.set_xlabel('Time (s)')
        if i == 0:
            ax.text(-0.16, 1.08, '(b)', transform=ax.transAxes,
                    fontweight='bold')

    handles = [Line2D([], [], color='k', lw=1.6, label='Analytic (Lamb)')]
    handles += [Line2D([], [], color=colors[d], lw=1.0, ls=(0, (4, 2)),
                       label=f'SV-FDM, dx = {d:g} m') for d in grids]
    handles.append(Line2D([], [], color='0.5', lw=0.7, ls=':', label='$x/C_R$'))
    fig.legend(handles=handles, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, 0.001), frameon=False,
               handletextpad=0.5, columnspacing=1.4, handlelength=1.9)

    fig.savefig(path, dpi=600, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mu = RHO * VS ** 2
    green = GreenfunctionFreesurface(RHO, RHO * VP ** 2 - 2 * mu, mu)
    cr = green.cd / green.zetar
    print(f"Homogeneous half-space: Vp={VP:.0f}, Vs={VS:.0f} m/s, "
          f"Cr={cr:.2f} m/s")

    wanted = os.environ.get('PRESET')
    names = [wanted] if wanted else list(PRESETS)

    for name in names:
        cfg = PRESETS[name]
        wavelength = cr / (2.5 * cfg['fc'])
        print(f"\n[{name}] fc = {cfg['fc']:g} Hz, "
              f"shortest wavelength {wavelength:.1f} m")

        t_a, analytic = analytic_velocity(green, cfg)
        fd, scales = {}, {}
        for dx in cfg['grids']:
            fd[dx] = run_fd(dx, cfg)
            scales[dx] = fit_scale(*fd[dx], t_a, analytic, cfg['offsets'])

        print(f"  {'dx (m)':>7s} {'pts/lambda':>11s} " +
              ' '.join(f'{int(o):>8d} m' for o in cfg['offsets']))
        for dx in cfg['grids']:
            vz, t = fd[dx]
            corrs = [np.corrcoef(vz[:, j] * scales[dx],
                                 np.interp(t, t_a, analytic[o]))[0, 1]
                     for j, o in enumerate(cfg['offsets'])]
            print(f"  {dx:7.1f} {wavelength / dx:11.1f} " +
                  ' '.join(f'{c:10.4f}' for c in corrs))

        png = os.path.join(OUT_DIR, f'analytic_comparison_{name}.png')
        plot_preset(name, cfg, cr, t_a, analytic, fd, scales, png)
        print(f"  saved {png}")


if __name__ == '__main__':
    main()
