# Strain-velocity FDM

A JAX implementation of 2D P-SV elastic wave propagation on a staggered grid,
written in the **velocity-strain** formulation rather than the usual
velocity-stress one. The solver is differentiable end to end, which is what it
was built for: gradients with respect to `Vp`, `Vs` and `rho` come straight
from `jax.grad`, so it drops into full-waveform inversion without an adjoint
solver written by hand.

Features:

- 2nd, 4th and 8th order spatial accuracy
- C-PML absorbing boundaries with a stretching factor (`kappa`), which handles
  grazing incidence better than the `kappa = 1` variant
- Free surface or PML on all four sides
- Block checkpointing so long simulations fit in memory under reverse-mode AD
- Optional DAS gauge-length averaging for strain recordings

## Install

```bash
pip install "jax[cuda12]" numpy scipy matplotlib   # or plain jax for CPU
```

## Use

```python
import numpy as np, jax.numpy as jnp
from fwi import build_forward_fn, ricker_jax

nz, nx, dx, dt, nt, fc = 152, 500, 20.0, 2e-3, 3000, 6.0
vs  = np.full((nz, nx), 1500.0)
vp  = np.full((nz, nx), 2600.0)
rho = np.full((nz, nx), 2000.0)

run_shot = build_forward_fn(
    nz, nx, dx, dx, dt, nt, fc,
    pad=80,                       # PML width in grid points
    block_size=250,               # checkpoint block length
    rec_x=np.arange(nx), rec_z=np.zeros(nx, dtype=int),
    fd_order=8, free_surface=True)

wavelet = ricker_jax(jnp.arange(nt) * dt, fc, 1.5 / fc)
vx, vz, exx, ezz = run_shot(jnp.array(vs), jnp.array(vp), jnp.array(rho),
                            jnp.int32(nx // 2), jnp.int32(0), wavelet)
```

`run_shot` is jitted and differentiable: `jax.grad` over a misfit built from
its output gives the FWI gradient directly.

## Validation

`validation/` cross-checks the solver against SPECFEM2D on Marmousi and
against the analytic 2D Lamb solution on a homogeneous half-space.

```bash
python validation/build_specfem_case.py    # writes DATA/ for SPECFEM2D
./validation/run_this_example.sh           # needs a SPECFEM2D build
python validation/compare_vz.py            # dx = 5 m by default
python validation/compare_analytic.py      # no SPECFEM2D needed
                                          # (downloads the Lamb solver once)
```

Vertical velocity on Marmousi, `fc = 3 Hz`, 8th order, `dx = 5 m`, correlation
over all 500 receivers:

| configuration | rel. RMS | correlation | amplitude ratio |
| --- | --- | --- | --- |
| body waves (PML on all sides, source 600 m deep) | 0.021 | 0.9998 | 1.003 |
| surface waves (free surface, source at the surface) | 0.398 | 0.921 | 0.896 |

![Marmousi comparison](validation/results/specfem_vs_svfdm_vz_dx5_fc3_ord8_nofs.png)

Against the analytic Lamb solution, correlation improves monotonically with
grid refinement and depends only on how many grid points sample the shortest
wavelength:

| points per wavelength | nearest offset | farthest offset |
| --- | --- | --- |
| 3.5 | 0.781 | 0.006 |
| 7.1 | 0.948 | 0.653 |
| 14.2 | 0.989 | 0.918 |
| 21.2 | 0.998 | 0.979 |

Surface waves are the limiting case: they travel in the slowest layer, so at a
given spacing they are sampled about twice as coarsely as body waves, and they
do not spread geometrically, so they dominate the record at long offsets.
Refining the grid is the effective remedy — raising the FD order is not, since
4th and 8th order already agree with each other.

## Third-party code

The analytic Lamb solution is computed by
[lamb_2dhalf_surface](https://github.com/ktkimit/lamb_2dhalf_surface)
(Ki-Tae Kim, LGPL-3.0). It is not redistributed here — that licence would
otherwise apply to this repository — so `compare_analytic.py` downloads it
into `validation/external/` on first run and uses it unmodified, apart from
dropping the demonstration block at the end of the file so it can be imported.

## Reference

Marmousi models under `marmousi_models/` are derived from Marmousi2
(Martin, Wiley & Marfurt, 2006, *The Leading Edge* 25(2), 156-166).
