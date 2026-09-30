"""
Validate the JAX velocity-strain FD solver against SPECFEM2D on Marmousi.

Both codes run the same model, source and receivers. SPECFEM2D is executed
beforehand (run_this_example.sh, inputs from build_specfem_case.py); this
script runs the FD forward solver, aligns the two datasets and produces the
comparison figure.

Usage
-----
    DX=5 SPEC_OUT=results/padded_fc3_out \
         SPEC_LOG=results/solver_padded_fc3.log \
         SPEC_PAR=DATA_padded_fc3/Par_file python compare_vz.py

Environment variables
---------------------
    DX        FD grid spacing in metres (default 20, the model grid).
              Smaller values interpolate the model onto a finer grid; SPECFEM
              does not need to be re-run because its mesh (40 m elements with
              4th-order GLL, i.e. ~10 m point spacing) is the reference.
    FC        Source frequency; taken from the SPECFEM SOURCE file by default.
    SPEC_OUT  Directory holding Uz_file_single_v.bin.
    SPEC_LOG  Matching solver log (used for the simulation start time).
    SPEC_PAR  Matching Par_file (boundary condition, geometry, time step).

Author: Minh Nhat Tran
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CASE_DIR))
import jax.numpy as jnp
from forward import build_forward_fn, ricker_jax

MODEL_DIR = os.path.join(os.path.dirname(CASE_DIR), 'marmousi_models')
OUT_DIR = os.path.join(CASE_DIR, 'results')

MODEL_DX = 20.0                       # grid spacing of the stored model
DX = float(os.environ.get('DX', 5.0))
# Stored SPECFEM2D run to compare against. `reference/` ships with the
# repository so the comparison reproduces without a SPECFEM2D installation;
# set SPEC_OUT/SPEC_LOG/SPEC_PAR to use your own run instead.
CASE = os.environ.get('CASE', 'nofs')     # 'nofs' or 'fs'
REFERENCE = os.path.join(CASE_DIR, 'reference', f'specfem_vz_{CASE}.npz')
SPEC_OUT = os.environ.get('SPEC_OUT')
SPEC_LOG = os.environ.get('SPEC_LOG')
SPEC_PAR = os.environ.get('SPEC_PAR')

SRC_X = 2000.0                        # source position in metres
TMAX = 6.0
COMPONENT = 1                         # 0=vx, 1=vz, 2=exx, 3=ezz
FD_ORDER = 8
PML = 400.0                           # PML width in metres (80 points at dx = 5 m)
BLOCK = 250
OFFSETS = [1000, 2000, 5000, 6000]
FD_STABILITY = {2: 1.0,
                4: 9 / 8 + 1 / 24,
                8: 1225 / 1024 + 245 / 3072 + 49 / 5120 + 5 / 7168}


# ── input parsing ───────────────────────────────────────────────────────────

def read_par(path, key):
    """Return the value of `key` from a SPECFEM Par_file / SOURCE file."""
    for line in open(path):
        if line.strip().startswith(key):
            return line.split('=', 1)[1].split('#')[0].strip()
    raise KeyError(f'{key} not found in {path}')


def specfem_t0(log_path):
    """Simulation start time printed by SPECFEM (negative, source peak at 0)."""
    for line in open(log_path):
        if 'simulation start time t0' in line:
            return float(line.split('=')[1].split()[0])
    raise RuntimeError(f'start time not found in {log_path}')


def load_specfem(out_dir, nrec):
    """Read the binary vertical-velocity seismograms as (nt, nrec)."""
    raw = np.fromfile(os.path.join(out_dir, 'Uz_file_single_v.bin'),
                      dtype=np.float32)
    nt = raw.size // nrec
    return raw[:nrec * nt].reshape(nrec, nt).T, nt


def load_reference(path):
    """Read a stored SPECFEM2D run shipped with the repository.

    Traces are held as float16 scaled to unit peak, which keeps the file small
    while staying three orders of magnitude more accurate than the differences
    being measured.
    """
    d = np.load(path)
    return (d['vz'].astype(np.float64) * float(d['scale']),
            float(d['dt']), float(d['t0']))


# geometry of the two shipped reference runs (see build_specfem_case.py)
REFERENCE_GEOMETRY = {
    'nofs': dict(free_surface=False, pml=400.0, rec_depth=200.0,
                 src_depth=600.0, x_first_rec=400.0, fc=3.0, dt=4.28e-4),
    'fs':   dict(free_surface=True, pml=400.0, rec_depth=0.0,
                 src_depth=0.0, x_first_rec=400.0, fc=3.0, dt=4.28e-4),
}


def read_geometry(par):
    """Boundary condition and source/receiver depths of the SPECFEM run."""
    data_dir = os.path.dirname(par)
    free_surface = read_par(par, 'absorbtop').startswith('.false')
    n_elem = int(read_par(par, 'NELEM_PML_THICKNESS'))
    elem = float(read_par(par, 'xmax').replace('d0', '')) / int(read_par(par, 'nx'))
    pml = n_elem * elem

    # SPECFEM measures z upward; the physical top lies below any padding layer
    info = open(os.path.join(data_dir,
                             'tomography_model_mine.xyz.bin.info')).read().splitlines()
    i = next(k for k, s in enumerate(info) if s.startswith('#origin_x'))
    z_top = float(info[i + 1].split()[3]) - (0.0 if free_surface else pml)

    return dict(
        free_surface=free_surface,
        pml=pml,
        rec_depth=z_top - float(read_par(par, 'zfin')),
        src_depth=z_top - float(read_par(os.path.join(data_dir, 'SOURCE'), 'zs')),
        x_first_rec=float(read_par(par, 'xdeb')),
        fc=float(os.environ.get(
            'FC', read_par(os.path.join(data_dir, 'SOURCE'), 'f0'))),
        dt=float(read_par(par, 'DT').replace('d', 'e').replace('D', 'e')),
    )


# ── model handling ──────────────────────────────────────────────────────────

def load_model(dx):
    """Load Marmousi and, if dx < MODEL_DX, interpolate onto a finer grid.

    scipy.ndimage.zoom is avoided here: it does not preserve the endpoint
    (500 columns become 1000 instead of 999), which stretches the model and
    shifts every physical coordinate. Interpolating on the true coordinate
    axes keeps receiver positions aligned with SPECFEM at any dx.
    """
    vp = np.load(os.path.join(MODEL_DIR, 'vp_true.npy'))
    vs = np.load(os.path.join(MODEL_DIR, 'vs_true.npy'))
    rho = np.load(os.path.join(MODEL_DIR, 'rho_true.npy'))
    nx_ref = vs.shape[1]

    if dx != MODEL_DX:
        nz0, nx0 = vs.shape
        x0, z0 = np.arange(nx0) * MODEL_DX, np.arange(nz0) * MODEL_DX
        xn = np.arange(0.0, x0[-1] + 1e-9, dx)
        zn = np.arange(0.0, z0[-1] + 1e-9, dx)

        def regrid(a):
            rows = np.array([np.interp(xn, x0, r) for r in a])
            return np.array([np.interp(zn, z0, c) for c in rows.T]).T

        vp, vs, rho = regrid(vp), regrid(vs), regrid(rho)

    return vp, vs, rho, nx_ref


# ── comparison ──────────────────────────────────────────────────────────────

def align(fd, spec, t_fd, t_spec):
    """Resample SPECFEM onto the FD time axis and match the amplitude scale.

    A single global factor accounts for the different source normalisations.
    It is the median of the per-trace peak ratios rather than a least-squares
    fit: most samples are low-amplitude coda where the two codes differ, and
    a least-squares fit over those biases the factor by an order of magnitude.
    """
    nt, nrec = len(t_fd), spec.shape[1]
    resampled = np.empty((nt, nrec))
    for i in range(nrec):
        resampled[:, i] = interp1d(t_spec, spec[:, i], bounds_error=False,
                                   fill_value=0.0)(t_fd)

    threshold = 0.05 * np.abs(resampled).max()
    ratios = [np.abs(fd[:, i]).max() / np.abs(resampled[:, i]).max()
              for i in range(nrec) if np.abs(resampled[:, i]).max() > threshold]
    scale = float(np.median(ratios))
    return resampled * scale, scale


def trace_metrics(a, b, dt, max_lag=0.5):
    """Relative RMS, correlation, best-fit lag, and correlation after shifting.

    The lag search keeps only positive correlation peaks within max_lag:
    taking |xcorr| instead locks onto the polarity-reversed peak half a cycle
    away at far offsets, which reports a spurious -0.95 as a good match.
    """
    rms = np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-30)
    corr = float(np.corrcoef(a, b)[0, 1])

    xa, xb = a - a.mean(), b - b.mean()
    cc = np.correlate(xa, xb, 'full') / (np.linalg.norm(xa) * np.linalg.norm(xb) + 1e-300)
    lags = (np.arange(cc.size) - (len(xa) - 1)) * dt
    window = np.abs(lags) <= max_lag
    k = int(np.flatnonzero(window)[np.argmax(cc[window])])
    return rms, corr, float(lags[k]), float(cc[k])


# ── figure ──────────────────────────────────────────────────────────────────

def plot(vs, fd, spec, t, locs, geom, dx, nx_ref, path):
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })

    n_rows = (len(locs) + 1) // 2
    x_km = nx_ref * MODEL_DX / 1000.0
    nz = vs.shape[0]

    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(2 + n_rows, 2,
                          height_ratios=[0.82, 1.15] + [0.62] * n_rows,
                          hspace=0.55, wspace=0.20,
                          left=0.085, right=0.945, top=0.985, bottom=0.085)

    # (a) model with acquisition geometry
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(vs, cmap='jet', aspect='auto',
                   extent=[0, x_km, nz * dx / 1000, 0])
    ax.plot(SRC_X / 1000, geom['src_depth'] / 1000, marker='*', ms=13,
            mfc='r', mec='k', mew=0.6, ls='none')
    ax.plot([l * MODEL_DX / 1000 for l in locs],
            [geom['rec_depth'] / 1000] * len(locs), marker='v', ms=6,
            mfc='w', mec='k', mew=0.8, ls='none', clip_on=False, zorder=5)
    ax.set(xlabel='Distance (km)', ylabel='Depth (km)', xlim=(0, x_km))
    cb = plt.colorbar(im, cax=ax.inset_axes([1.012, 0.0, 0.018, 1.0]))
    cb.set_label('$V_S$ (m/s)')
    cb.ax.tick_params(labelsize=7)
    ax.text(-0.075, 1.06, '(a)', transform=ax.transAxes, fontweight='bold')

    # (b, c) shot gathers
    clip = 1e-2 * np.abs(fd).max()
    for k, (data, title) in enumerate([(fd, 'SV-FDM (this study)'),
                                       (spec, 'SPECFEM2D')]):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(data, cmap='gray', aspect='auto', vmin=-clip, vmax=clip,
                  extent=[0, x_km, t[-1], t[0]])
        ax.set(xlabel='Distance (km)', xlim=(0, x_km), ylim=(t[-1], t[0]))
        ax.set_title(title, pad=4)
        if k == 0:
            ax.set_ylabel('Time (s)')
        else:
            ax.tick_params(labelleft=False)
        ax.text(-0.16 if k == 0 else -0.06, 1.06, f'({"bc"[k]})',
                transform=ax.transAxes, fontweight='bold')

    # (d) waveform comparison
    for i, (offset, loc) in enumerate(zip(OFFSETS, locs)):
        ax = fig.add_subplot(gs[2 + i // 2, i % 2])
        ax.plot(t, spec[:, loc], color='k', lw=1.0)
        ax.plot(t, fd[:, loc], color='r', lw=1.0, ls=(0, (4, 2)))
        ax.set_xlim(t[0], t[-1])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(7)
        ax.text(0.975, 0.90, f'{offset / 1000:g} km', transform=ax.transAxes,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.22', fc='w', ec='0.6', lw=0.5))
        if i % 2 == 0:
            ax.set_ylabel('$v_z$')
        if i // 2 == n_rows - 1:
            ax.set_xlabel('Time (s)')
        else:
            ax.tick_params(labelbottom=False)
        if i == 0:
            ax.text(-0.16, 1.10, '(d)', transform=ax.transAxes,
                    fontweight='bold')

    fig.legend(handles=[
        Line2D([], [], marker='*', ms=12, mfc='r', mec='k', mew=0.6,
               ls='none', label='Source'),
        Line2D([], [], marker='v', ms=6, mfc='w', mec='k', mew=0.8,
               ls='none', label='Receivers'),
        Line2D([], [], color='k', lw=1.0, label='SPECFEM2D'),
        Line2D([], [], color='r', lw=1.0, ls=(0, (4, 2)), label='SV-FDM'),
    ], loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.001), frameon=False,
        handletextpad=0.5, columnspacing=1.8, handlelength=1.9)

    fig.savefig(path, dpi=600, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    using_reference = SPEC_PAR is None
    geom = (REFERENCE_GEOMETRY[CASE] if using_reference
            else read_geometry(SPEC_PAR))
    fc = geom['fc']
    print(f"SPECFEM: f0={fc} Hz, dt={geom['dt'] * 1e3:.4f} ms, "
          f"free_surface={geom['free_surface']}")
    print(f"  source depth {geom['src_depth']:.0f} m, "
          f"receiver depth {geom['rec_depth']:.0f} m")

    vp, vs, rho, nx_ref = load_model(DX)
    nz, nx = vs.shape
    step = int(round(MODEL_DX / DX))
    pad = int(round(PML / DX))
    print(f"  FD grid dx={DX:g} m -> {nz} x {nx}, PML {pad} points")

    # forward modelling on the FD grid
    dt = 0.9 / (FD_STABILITY[FD_ORDER] * vp.max() * np.sqrt(2) / DX)
    nt = int(TMAX / dt)
    src_ix = int(round(SRC_X / DX))
    src_iz = int(round(geom['src_depth'] / DX))
    rec_iz = int(round(geom['rec_depth'] / DX))

    # record only on the model columns; a receiver at every fine-grid column
    # would hold nt x nx samples, several GB at dx = 1 m
    rec_ix = np.arange(nx_ref) * step
    run_shot = build_forward_fn(nz, nx, DX, DX, dt, nt, fc, pad, BLOCK,
                                rec_ix, np.full(nx_ref, rec_iz, dtype=int),
                                fd_order=FD_ORDER, component=COMPONENT,
                                free_surface=geom['free_surface'])
    wavelet = ricker_jax(jnp.arange(nt) * dt, fc, 1.5 / fc)
    fd = np.asarray(run_shot(jnp.array(vs), jnp.array(vp), jnp.array(rho),
                             jnp.int32(src_ix), jnp.int32(src_iz),
                             wavelet)[COMPONENT])
    print(f"  FD dt={dt * 1e3:.4f} ms, nt={nt}, gather {fd.shape}")

    # align both datasets on a common time origin (the source peak)
    if using_reference:
        spec_raw, dt_spec, t0_spec = load_reference(REFERENCE)
        nt_spec = spec_raw.shape[0]
        print(f"  reference run: {os.path.basename(REFERENCE)}")
    else:
        spec_raw, nt_spec = load_specfem(SPEC_OUT, nx_ref)
        dt_spec, t0_spec = geom['dt'], specfem_t0(SPEC_LOG)
    t = np.arange(nt) * dt - 1.5 / fc
    t_spec = t0_spec + np.arange(nt_spec) * dt_spec
    spec, scale = align(fd, spec_raw, t, t_spec)
    print(f"  amplitude scale (SPECFEM -> FD) = {scale:.4f}")

    # per-receiver metrics
    src_col = int(round(SRC_X / MODEL_DX))
    locs = [src_col + o // int(MODEL_DX) for o in OFFSETS]
    locs = [l for l in locs if l < nx_ref]

    print(f"\n  {'offset':>8s} {'rel_rms':>9s} {'corr':>8s} "
          f"{'lag(ms)':>9s} {'corr@lag':>9s}")
    for offset, loc in zip(OFFSETS, locs):
        rms, corr, lag, corr_lag = trace_metrics(fd[:, loc], spec[:, loc], dt)
        print(f"  {offset:8d} {rms:9.4f} {corr:+8.4f} "
              f"{lag * 1e3:+9.1f} {corr_lag:+9.4f}")

    # global metrics; exclude any receiver that sits inside the SPECFEM PML
    n_edge = max(0, int(np.ceil((geom['pml'] - geom['x_first_rec']) / MODEL_DX)))
    mask = np.zeros(nx_ref, dtype=bool)
    mask[n_edge:nx_ref - n_edge if n_edge else nx_ref] = True
    mask[src_col] = False
    a, b = fd[:, mask].ravel(), spec[:, mask].ravel()
    print(f"  {'GLOBAL':>8s} "
          f"{np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)):9.4f} "
          f"{np.corrcoef(a, b)[0, 1]:+8.4f}")

    tag = f"dx{DX:g}_fc{fc:g}_ord{FD_ORDER}"
    tag += '_fs' if geom['free_surface'] else '_nofs'
    png = os.path.join(OUT_DIR, f'specfem_vs_svfdm_vz_{tag}.png')
    plot(vs, fd, spec, t, locs, geom, DX, nx_ref, png)
    np.savez(os.path.join(OUT_DIR, f'gathers_{tag}.npz'),
             fd=fd, spec=spec, t=t, dt=dt, scale=scale)
    print(f"\nSaved {png}")


if __name__ == '__main__':
    main()
