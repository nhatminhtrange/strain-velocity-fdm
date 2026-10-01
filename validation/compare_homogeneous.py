"""
Homogeneous half-space: SPECFEM2D, the analytic Lamb solution and SV-FDM.

A vertical Ricker force on the free surface of a half-space (Vp 2000 m/s,
Vs 1155 m/s, rho 2000 kg/m^3), recorded on the surface, for fc = 5 Hz.
SPECFEM2D uses 20 m spectral elements, SV-FDM a 1 m grid (FD_DX). The
setup is described in common.py.

Nothing is rescaled. The only sign change is SPECFEM's vx (SPECFEM_SIGN in
common.py).

Usage:
    CUDA_VISIBLE_DEVICES=0 python compare_homogeneous.py

Writes results/homogeneous_{vz,vx}.png and prints relative RMS errors
against the analytic solution.

The analytic Green's function comes from lamb_2dhalf_surface (Ki-Tae Kim,
LGPL-3.0, https://github.com/ktkimit/lamb_2dhalf_surface). It is not
redistributed here: fetch_lamb_solver() downloads it into external/ on first
use and it is used unmodified, apart from dropping the demonstration block at
the end of the file so that it can be imported.

The reference implementation convolves with the Ricker wavelet over [0, t].
Late in the record that interval is much longer than the wavelet and its
adaptive quadrature misses it, giving errors of several per cent in uz.
CompactRicker below integrates over the wavelet support only; the Green's
function is used unchanged.

Author: Minh Nhat Tran
"""
import os
import sys

import numpy as np
from scipy.integrate import quad
from common import (COMPONENTS, FC, FD_DX, HALF_SPACE, OUT_DIR, SPECFEM_SIGN,
                    T_MAX, comparison_figure, fd_gather, load_reference, receiver_index, rel_rms)

# ── analytic Lamb solution ──────────────────────────────────────────────────

UPSTREAM = ('https://raw.githubusercontent.com/ktkimit/lamb_2dhalf_surface/'
            'master/lamb2d_freesurface.py')
EXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external')


def fetch_lamb_solver():
    path = os.path.join(EXT_DIR, 'lamb2d_freesurface.py')
    if not os.path.exists(path):
        from urllib.request import urlopen
        os.makedirs(EXT_DIR, exist_ok=True)
        print(f'fetching {UPSTREAM}')
        text = urlopen(UPSTREAM, timeout=60).read().decode()
        marker = "# Young's modulus"
        if marker in text:
            text = text[:text.index(marker)].rstrip() + '\n'
        with open(path, 'w') as f:
            f.write(text)
        with open(os.path.join(EXT_DIR, 'README.md'), 'w') as f:
            f.write('# Third-party code (not part of this repository)\n\n'
                    'lamb2d_freesurface.py is downloaded from\n'
                    'https://github.com/ktkimit/lamb_2dhalf_surface\n'
                    '(Ki-Tae Kim, LGPL-3.0). Only the demonstration block at\n'
                    'the end of the file is removed so it can be imported.\n')
    sys.path.insert(0, EXT_DIR)


fetch_lamb_solver()
from lamb2d_freesurface import GreenfunctionFreesurface, Ricker  # noqa: E402


class CompactRicker(Ricker):
    """Ricker convolution integrated over the wavelet support only.

    Outside delay +- 2/fc the Ricker is below 1e-17 of its peak, so
    restricting the integral to that window changes nothing but the
    robustness of the quadrature.

    The displacements are of order 1e-12, far below quad's default absolute
    tolerance (1.5e-8), which would let it stop on a wrong estimate; only a
    relative tolerance is used.
    """

    def integration_convolution(self, x, t):
        g = self.green
        if g.t2tau(x, t) < 1.0:
            return (0.0, 0.0)
        half = 2.0 / self.freq
        a = max(0.0, self.delay - half)
        b = min(t - g.tau2t(x, 1.0), self.delay + half)
        if b <= a:
            return (0.0, 0.0)

        tt_k = t - g.tau2t(x, g.k)
        tt_r = t - g.tau2t(x, g.zetar)
        points = [p for p in (tt_k, tt_r) if a < p < b] or None
        fu = lambda tt: self.evaluate_convolution(x, t, tt)[0]
        fw = lambda tt: self.evaluate_convolution(x, t, tt)[1]
        tol = dict(limit=200, epsabs=0.0, epsrel=1e-10)

        # ux: regular integral plus the Rayleigh-pole contribution
        u = quad(fu, a, b, points=points, **tol)[0]
        if 0.0 <= tt_r <= t:
            u += self.evaluate(tt_r) * x / g.cd * g.evaluate(x, g.zetar)[0]

        # uz: principal value across the Rayleigh pole
        if a < tt_r < b:
            c = min(tt_k, b)
            w = quad(lambda tt: fw(tt) * (tt - tt_r), a, c,
                     weight='cauchy', wvar=tt_r, **tol)[0]
            if c < b:
                w += quad(fw, c, b, **tol)[0]
        else:
            w = quad(fw, a, b, points=points, **tol)[0]
        return (u, w)


def _remove_outliers(d):
    """Replace isolated samples where the quadrature failed to converge."""
    d = d.copy()
    floor = 1e-4 * np.abs(d).max()
    for _ in range(3):
        d2 = np.abs(np.diff(d, 2))
        side = np.maximum(np.r_[d2[2:], 0, 0], np.r_[0, 0, d2[:-2]])
        bad = np.flatnonzero(d2 > 4 * side + floor) + 1
        if bad.size == 0:
            break
        d[bad] = 0.5 * (d[bad - 1] + d[bad + 1])
    return d


def half_space(vp, vs, rho):
    mu = rho * vs ** 2
    return GreenfunctionFreesurface(rho, rho * vp ** 2 - 2 * mu, mu)


def surface_velocity(green, fc, offset, t):
    """(vx, vz) at `offset` for a unit Ricker line force peaking at t = 0."""
    delay = 1.5 / fc
    ricker = CompactRicker(green, 1.0, fc, delay)
    ts = t + delay
    u = np.array([ricker.integration_convolution(offset, ti) if ti > 0
                  else (0.0, 0.0) for ti in ts])
    return tuple(np.gradient(_remove_outliers(u[:, k]), ts) for k in range(2))


# ── comparison ──────────────────────────────────────────────────────────────

OFFSETS = [1000, 2000, 3000, 5000]
LAMB_DT = 2e-3           # analytic samples; all traces compared on this axis


def lamb_traces(green, fc, offsets, t):
    """Analytic vx, vz at each offset, cached in results/."""
    cache = os.path.join(OUT_DIR, f'lamb_fc{fc:g}.npz')
    if os.path.exists(cache):
        d = np.load(cache)
        if (d['t'].shape == t.shape and np.allclose(d['t'], t)
                and list(d['offsets']) == list(offsets)):
            return {'vx': d['vx'], 'vz': d['vz']}
    print(f"  Lamb fc={fc:g} Hz at offsets {offsets} ...")
    vx, vz = zip(*(surface_velocity(green, fc, o, t) for o in offsets))
    out = {'vx': np.array(vx), 'vz': np.array(vz)}
    np.savez(cache, t=t, offsets=offsets, **out)
    return out


def collect(green):
    """Common time axis and Lamb / SPECFEM / FD traces at OFFSETS."""
    ref = load_reference('homogeneous', FC)
    fd = fd_gather('homogeneous', FC, ref)
    step = max(1, int(round(LAMB_DT / (ref['t'][1] - ref['t'][0]))))
    t = ref['t'][::step]
    idx = [receiver_index(ref, o) for o in OFFSETS]
    return dict(
        t=t, src_x=ref['src_x'],
        lamb=lamb_traces(green, FC, OFFSETS, t),
        spec={c: np.array([SPECFEM_SIGN[c] * ref[c][::step, i] for i in idx])
              for c in COMPONENTS},
        fd={c: np.array([fd[c][::step, i] for i in idx]) for c in COMPONENTS})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    green = half_space(**{k: HALF_SPACE[k] for k in ('vp', 'vs', 'rho')})
    r = collect(green)

    print(f"\nRelative RMS error against Lamb, fc = {FC:g} Hz "
          f"(whole trace, no scaling)")
    print(f"{'comp':>4s} {'code':>8s} " +
          ' '.join(f'{o / 1000:>6g}km' for o in OFFSETS))
    for c in COMPONENTS:
        for code in ('spec', 'fd'):
            errs = [rel_rms(r[code][c][i], r['lamb'][c][i])
                    for i in range(len(OFFSETS))]
            print(f"{c:>4s} {code:>8s} " + ' '.join(f'{e:8.4f}' for e in errs))

    for c in COMPONENTS:
        path = os.path.join(OUT_DIR, f'homogeneous_{c}.png')
        comparison_figure(
            'homogeneous', r['src_x'], OFFSETS, r['t'],
            [('Analytic (Lamb)', r['lamb'][c], dict(color='k', lw=2.2)),
             ('SPECFEM2D', r['spec'][c], dict(color='tab:blue', lw=1.1)),
             (f'SV-FDM, grid size = {FD_DX:g} m', r['fd'][c],
              dict(color='tab:red', lw=1.1, ls=(0, (3, 2))))],
            f'$v_{c[1]}$', path, T_MAX)
        print(f"Saved {path}")


if __name__ == '__main__':
    main()
