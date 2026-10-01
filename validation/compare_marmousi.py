"""
Marmousi with a free surface: SPECFEM2D against SV-FDM.

A vertical Ricker force at x = 2 km on the free surface, 500 receivers on
the surface, fc = 5 Hz. SPECFEM2D uses 20 m spectral elements, SV-FDM a
1 m grid (FD_DX). The setup is described in common.py. Nothing is rescaled.

Usage:
    CUDA_VISIBLE_DEVICES=0 python compare_marmousi.py

Writes results/marmousi_{vz,vx}.png, the same comparison on coarser FD grids
(CONV_DX) in results/marmousi_convergence_{vz,vx}.png, and prints per-trace and whole-gather
relative RMS errors. SPECFEM's vx is sign-flipped (SPECFEM_SIGN in common.py).

Author: Minh Nhat Tran
"""
import os

import numpy as np

from common import (COMPONENTS, FC, FD_DX, SPECFEM_SIGN, OUT_DIR, T_MAX, comparison_figure, fd_gather,
                    load_reference, receiver_index, rel_rms)

OFFSETS = [1000, 2000, 3000, 5000]
CONV_DX = [10.0, 5.0, FD_DX]          # grids of the convergence figure
CONV_STYLE = {10.0: dict(color='tab:blue', lw=1.0),
              5.0: dict(color='tab:orange', lw=1.0)}
MIN_OFFSET = 200.0       # receivers closer to the source are left out of the
                         # whole-gather metrics (near-field singularity)


def gather_rms(ref, fd, c):
    keep = np.abs(ref['rec_x'] - ref['src_x']) >= MIN_OFFSET
    win = ref['t'] <= T_MAX
    return rel_rms(fd[c][np.ix_(win, keep)], ref[c][np.ix_(win, keep)])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = load_reference('marmousi', FC)
    for c in COMPONENTS:
        ref[c] = SPECFEM_SIGN[c] * ref[c]
    fd = fd_gather('marmousi', FC, ref)
    idx = [receiver_index(ref, o) for o in OFFSETS]
    win = ref['t'] <= T_MAX

    print(f"\nSV-FDM vs SPECFEM2D, fc = {FC:g} Hz, no scaling "
          f"(whole gather: offsets >= {MIN_OFFSET:g} m, t <= {T_MAX:g} s)")
    for c in COMPONENTS:
        per = '  '.join(f"{o / 1000:g}km {rel_rms(fd[c][win, k], ref[c][win, k]):.3f}"
                        for o, k in zip(OFFSETS, idx))
        print(f"  {c}: gather rel. RMS {gather_rms(ref, fd, c):.3f} | "
              f"trace rel. RMS: {per}")

    for c in COMPONENTS:
        path = os.path.join(OUT_DIR, f'marmousi_{c}.png')
        comparison_figure(
            'marmousi', ref['src_x'], OFFSETS, ref['t'],
            [('SPECFEM2D', ref[c][:, idx].T, dict(color='k', lw=1.5)),
             (f'SV-FDM, grid size = {FD_DX:g} m', fd[c][:, idx].T,
              dict(color='tab:red', lw=1.0, ls=(0, (3, 2))))],
            f'$v_{c[1]}$', path, T_MAX)
        print(f"Saved {path}")

    # convergence: the same comparison on coarser FD grids
    fds = {dx: fd if dx == FD_DX else fd_gather('marmousi', FC, ref, dx)
           for dx in CONV_DX}
    print(f"\nConvergence, trace rel. RMS vs SPECFEM2D (t <= {T_MAX:g} s)")
    print(f"  {'comp':>4s} {'dx':>5s} " +
          ' '.join(f'{o / 1000:>6g}km' for o in OFFSETS) + '   gather')
    for c in COMPONENTS:
        for dx in CONV_DX:
            errs = [rel_rms(fds[dx][c][win, k], ref[c][win, k]) for k in idx]
            print(f"  {c:>4s} {dx:5g} " + ' '.join(f'{e:8.3f}' for e in errs) +
                  f'  {gather_rms(ref, fds[dx], c):7.3f}')

    for c in COMPONENTS:
        curves = [('SPECFEM2D', ref[c][:, idx].T, dict(color='k', lw=1.8))]
        curves += [(f'SV-FDM, grid size = {dx:g} m', fds[dx][c][:, idx].T,
                    CONV_STYLE.get(dx, dict(color='tab:red', lw=1.0,
                                            ls=(0, (3, 2)))))
                   for dx in CONV_DX]
        path = os.path.join(OUT_DIR, f'marmousi_convergence_{c}.png')
        comparison_figure('marmousi', ref['src_x'], OFFSETS, ref['t'], curves,
                          f'$v_{c[1]}$', path, T_MAX)
        print(f"Saved {path}")


if __name__ == '__main__':
    main()
