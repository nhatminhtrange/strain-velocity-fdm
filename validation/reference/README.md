# Stored SPECFEM2D reference runs

Shot gathers from SPECFEM2D, stored so the comparisons reproduce without a
SPECFEM2D installation. Every file holds all 500 receivers (one per 20 m
model column, x = 0 ... 9980 m) on SPECFEM's own time axis:
`t = t0 + arange(nt) * dt`, with the Ricker peak at t = 0.

Traces are float16 normalised to unit peak; multiply by the matching scale to
recover particle velocity (m/s). The quantisation error is about 2e-4
relative RMS per trace.

| File | Model | fc | SPECFEM element | Components |
| --- | --- | --- | --- | --- |
| `specfem_fc5_homogeneous_fs.npz` | half-space Vp 2000, Vs 1155, rho 2000 | 5 Hz | 20 m | vx, vz |
| `specfem_fc5_marmousi_fs.npz` | Marmousi | 5 Hz | 20 m | vx, vz |

All runs: vertical point force (Ricker) at x = 2000 m on the free surface of
a 10 x 3 km domain, receivers on the surface, 400 m PML outside the domain on
the sides and bottom, 4th-order GLL, 16 MPI processes. The inputs come from
`../build_specfem_case.py`. Against
the analytic Lamb solution, the homogeneous runs are within 0.2 % relative
RMS at every offset from 1 to 5 km.

## Keys

* `vz`, `scale` — vertical velocity, (nt, 500) float16, and its scale factor.
  `common.load_reference()` reads these.
* `vx`, `scale_vx` — horizontal velocity.
  `scale_vz` duplicates `scale`.
* `dt`, `t0` — SPECFEM time step and start time.
* Every file also carries the geometry: `fc`, `free_surface`, `pml`, `src_x`,
  `src_depth`, `rec_x`, `rec_depth` (physical coordinates, depth positive
  down), `elem_size`, `model`, and the exact `par_file` and `source_file`
  text used for the run.

## Sign convention

SPECFEM measures z upward and the force points up (+z). The FD solver in
`forward.py` measures z downward with the force pointing down. On the
homogeneous half-space, SPECFEM vz agrees with both the analytic Lamb
solution and the FD solver as stored, while SPECFEM vx has the opposite
polarity (same amplitude and timing).

```python
import numpy as np
d = np.load('specfem_fc5_homogeneous_fs.npz')
vz = d['vz'].astype(float) * float(d['scale'])
vx = d['vx'].astype(float) * float(d['scale_vx'])
t = float(d['t0']) + np.arange(vz.shape[0]) * float(d['dt'])
```
