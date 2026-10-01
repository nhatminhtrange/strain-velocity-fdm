"""
Shared pieces of the SPECFEM2D / SV-FDM comparisons.

Both comparisons use the same setup: a vertical Ricker force at x = 2 km on
the free surface of a 10 km x 3 km domain, 500 receivers on the surface (one
per 20 m model column), and PML outside the domain on the sides and bottom.
Only the material changes (homogeneous half-space or Marmousi).

SPECFEM gathers are read from reference/specfem_fc<fc>_<model>_fs.npz (see
reference/README.md). The FD solver is run on the same domain at spacing
FD_DX, the model being interpolated bilinearly from the 20 m grid, which is
also what SPECFEM does with its tomography file. FD gathers are cached in
results/ because a 1 m grid takes several minutes per run; they can be
computed ahead of time, one per GPU:

    CUDA_VISIBLE_DEVICES=0 python common.py homogeneous 5

All codes use a unit line force and the same Ricker wavelet, so nothing is
rescaled. Traces share one time axis with the source peak at t = 0.

Author: Minh Nhat Tran
"""
import os
import sys

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(VALIDATION_DIR, 'reference')
OUT_DIR = os.path.join(VALIDATION_DIR, 'results')
MODEL_DIR = os.path.join(os.path.dirname(VALIDATION_DIR), 'marmousi_models')
sys.path.insert(0, os.path.dirname(VALIDATION_DIR))

FC = float(os.environ.get('FC', 5.0))     # Ricker peak frequency (Hz)
FD_DX = float(os.environ.get('FD_DX', 1.0))
FD_ORDER = 8
FD_STABILITY = 1225 / 1024 + 245 / 3072 + 49 / 5120 + 5 / 7168   # 8th order
MODEL_DX = 20.0
PML = 400.0                  # FD absorbing layer, same thickness as SPECFEM's
BLOCK = 250

HALF_SPACE = dict(vp=2000.0, vs=1155.0, rho=2000.0)

# SPECFEM measures z upward and its force points up; the FD solver and Lamb
# have z and the force pointing down. That flips the horizontal component
# and leaves the vertical one unchanged.
SPECFEM_SIGN = {'vz': 1.0, 'vx': -1.0}
COMPONENTS = ('vz', 'vx')

# grids of the convergence figures; the finest is FD_DX
CONV_DX = [10.0, 5.0, 2.0, FD_DX]
CONV_STYLE = {10.0: dict(color='tab:green', lw=1.0),
              5.0: dict(color='tab:orange', lw=1.0),
              2.0: dict(color='tab:purple', lw=1.0)}
FINE_STYLE = dict(color='tab:red', lw=1.0, ls=(0, (3, 2)))


def conv_style(dx):
    return CONV_STYLE.get(dx, FINE_STYLE)


T_MAX = 5.5                  # end of the plotted / compared record (s)

PLOT_STYLE = {
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8.5,
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'savefig.dpi': 300,
}


def load_model(name):
    """(vp, vs, rho) on the 20 m model grid, row 0 at the surface."""
    vs = np.load(os.path.join(MODEL_DIR, 'vs_true.npy')).astype(np.float64)
    if name == 'homogeneous':
        return tuple(np.full_like(vs, HALF_SPACE[k]) for k in ('vp', 'vs', 'rho'))
    vp = np.load(os.path.join(MODEL_DIR, 'vp_true.npy')).astype(np.float64)
    rho = np.load(os.path.join(MODEL_DIR, 'rho_true.npy')).astype(np.float64)
    return vp, vs, rho


def load_reference(model, fc):
    """SPECFEM gather: t (source peak at 0), vz and vx in m/s, geometry.

    Read from reference/specfem_fc<fc>_<model>_fs.npz. The float16 traces are
    only multiplied back by their stored scale; nothing else is rescaled.
    """
    d = np.load(os.path.join(REFERENCE_DIR, f'specfem_fc{fc:g}_{model}_fs.npz'))
    ref = dict(rec_x=d['rec_x'].astype(np.float64), src_x=float(d['src_x']),
               elem_size=float(d['elem_size']))
    for comp in ('x', 'z'):
        ref[f'v{comp}'] = (d[f'v{comp}'].astype(np.float64)
                           * float(d[f'scale_v{comp}']))
    ref['t'] = float(d['t0']) + np.arange(ref['vz'].shape[0]) * float(d['dt'])
    return ref


def regrid(a, dx):
    """Bilinear interpolation of a 20 m model onto a dx grid, endpoints kept."""
    if dx == MODEL_DX:
        return a
    nz0, nx0 = a.shape
    x0, z0 = np.arange(nx0) * MODEL_DX, np.arange(nz0) * MODEL_DX
    xn = np.arange(0.0, x0[-1] + 1e-9, dx)
    zn = np.arange(0.0, z0[-1] + 1e-9, dx)
    rows = np.array([np.interp(xn, x0, r) for r in a])
    return np.array([np.interp(zn, z0, c) for c in rows.T]).T


def fd_gather(model, fc, ref, dx=FD_DX):
    """SV-FDM vx, vz at the SPECFEM receivers, resampled onto ref['t'].

    Cached in results/fd_<model>_fc<fc>_dx<dx>.npz.
    """
    cache = os.path.join(OUT_DIR, f'fd_{model}_fc{fc:g}_dx{dx:g}.npz')
    if os.path.exists(cache):
        d = np.load(cache)
        if d['t'].shape == ref['t'].shape and np.allclose(d['t'], ref['t']):
            return {'vx': d['vx'], 'vz': d['vz']}

    import jax.numpy as jnp
    from forward import build_forward_fn, ricker_jax

    vp, vs, rho = (regrid(a, dx) for a in load_model(model))
    nz, nx = vs.shape
    dt = 0.9 / (FD_STABILITY * vp.max() * np.sqrt(2) / dx)
    delay = 1.5 / fc
    nt = int(np.ceil((ref['t'][-1] + delay) / dt)) + 1
    pad = int(round(PML / dx))
    rec_ix = np.round(ref['rec_x'] / dx).astype(int)
    print(f"  FD {model} fc={fc:g} Hz: dx={dx:g} m, grid {nz} x {nx}, "
          f"dt={dt * 1e3:.4f} ms, nt={nt}")

    run_shot = build_forward_fn(nz, nx, dx, dx, dt, nt, fc, pad, BLOCK,
                                rec_ix, np.zeros(len(rec_ix), dtype=int),
                                fd_order=FD_ORDER, free_surface=True)
    wavelet = ricker_jax(jnp.arange(nt) * dt, fc, delay)
    out = run_shot(jnp.array(vs), jnp.array(vp), jnp.array(rho),
                   jnp.int32(int(round(ref['src_x'] / dx))), jnp.int32(0),
                   wavelet)
    t_fd = np.arange(nt) * dt - delay
    res = {}
    for k, comp in (('vx', 0), ('vz', 1)):
        g = np.asarray(out[comp])
        res[k] = np.stack([np.interp(ref['t'], t_fd, g[:, i])
                           for i in range(g.shape[1])], axis=1).astype(np.float32)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(cache, t=ref['t'], **res)
    return res


def rel_rms(a, ref):
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def receiver_index(ref, offset):
    return int(np.argmin(np.abs(ref['rec_x'] - ref['src_x'] - offset)))


def comparison_figure(model, src_x, offsets, t, curves, ylabel, path,
                      t_max=5.5, label_room=0.25):
    """Shared layout of both comparisons.

    (a) the Vs model with source and receivers, (b) one trace panel per
    offset on a 2 x 2 grid, offset labelled in the lower-left corner, and
    the legend below. `curves` is a list of (label, traces, line style),
    traces being (len(offsets), len(t)).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(PLOT_STYLE)
    km = 1000.0
    vs = load_model(model)[1]
    nz, nx = vs.shape
    vs_range = load_model('marmousi')[1]       # same colour scale for both

    fig = plt.figure(figsize=(7.2, 6.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.75], hspace=0.30)
    gs = outer[1].subgridspec(2, 2, hspace=0.30, wspace=0.12)

    # (a) model and geometry
    ax = fig.add_subplot(outer[0])
    im = ax.imshow(vs, cmap='viridis', aspect='auto',
                   vmin=vs_range.min(), vmax=vs_range.max(),
                   extent=(0, (nx - 1) * MODEL_DX / km,
                           (nz - 1) * MODEL_DX / km, 0))
    ax.plot(src_x / km, 0, marker='*', ms=13, mfc='r', mec='k', mew=0.6,
            ls='none', clip_on=False, zorder=5)
    ax.plot([(src_x + o) / km for o in offsets], [0] * len(offsets),
            marker='v', ms=7, mfc='w', mec='k', mew=0.8, ls='none',
            clip_on=False, zorder=5)
    ax.set(xlabel='Distance (km)', ylabel='Depth (km)')
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label('$V_S$ (m/s)')
    ax.text(-0.07, 1.04, '(a)', transform=ax.transAxes, fontweight='bold')

    # (b) traces on a 2 x 2 grid
    win = t <= t_max
    for i, o in enumerate(offsets):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        for z, (_, traces, style) in enumerate(curves):
            ax.plot(t[win], traces[i][win], zorder=z + 1, **style)
        ax.set_xlim(-0.3, t_max)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(7)
        lo, hi = ax.get_ylim()                 # keep the label clear of traces
        ax.set_ylim(lo - label_room / (1 - label_room) * (hi - lo), hi)
        ax.text(0.02, 0.04, f'{o / km:g} km', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.22', fc='w', ec='0.6', lw=0.5))
        if i >= 2:
            ax.set_xlabel('Time (s)')
        else:
            ax.tick_params(labelbottom=False)
        if i % 2 == 0:
            ax.set_ylabel(ylabel)
        if i == 0:
            ax.text(-0.17, 1.08, '(b)', transform=ax.transAxes,
                    fontweight='bold')

    handles = [Line2D([], [], marker='*', ms=11, mfc='r', mec='k', mew=0.6,
                      ls='none', label='Source'),
               Line2D([], [], marker='v', ms=6, mfc='w', mec='k', mew=0.8,
                      ls='none', label='Receivers')]
    handles += [Line2D([], [], label=label, **style)
                for label, _, style in curves]
    ncol = len(handles) if len(handles) <= 5 else 4
    rows = -(-len(handles) // ncol)
    fig.legend(handles=handles, loc='lower center', ncol=ncol,
               frameon=False, bbox_to_anchor=(0.5, -0.03 - 0.035 * (rows - 1)),
               columnspacing=1.2, handletextpad=0.5)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # precompute one FD gather: python common.py <homogeneous|marmousi> <fc>
    model, fc = sys.argv[1], float(sys.argv[2])
    fd_gather(model, fc, load_reference(model, fc))
    print(f"  cached fd_{model}_fc{fc:g}_dx{FD_DX:g}.npz")
