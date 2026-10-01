# SPECFEM2D example input (Marmousi, fc = 5 Hz)

`DATA/` holds the exact SPECFEM2D input used for the Marmousi reference run:
`Par_file`, `SOURCE`, `STATIONS`, `interfaces.dat` and the model as a
tomography file. `../build_specfem_case.py` with its default settings
regenerates these files byte for byte.

Setup: Marmousi (10 x 3 km, 20 m grid) with a free surface on top and a
400 m PML on the sides and bottom; vertical Ricker force, fc = 5 Hz, at
x = 2000 m on the surface; 500 receivers on the surface every 20 m
(x = 0 ... 9980 m); 20 m spectral elements, 4th-order GLL;
dt = 2.14e-4 s, 28039 steps (about 6 s); 16 MPI processes.

## Run

From `validation/`:

    cp -r specfem_example/DATA .
    SPECFEM_ROOT=/path/to/specfem2d MPIRUN=mpirun ./run_this_example.sh > solver.log 2>&1

* Keep `solver.log`: the simulation start time t0 (the source peak is at
  t = 0) is printed there as "simulation start time t0".
* `NPROC = 16` in `DATA/Par_file`; change it to match your machine.
* Tested with SPECFEM2D v8.1.0-179-g24593fa5, built with OpenMPI.

## Output

`OUTPUT_FILES/Ux_file_single_v.bin` and `Uz_file_single_v.bin`: particle
velocity (m/s), float32, receiver-major:

```python
import numpy as np
nrec, dt, t0 = 500, 2.13988827e-04, -0.24   # t0 from solver.log
vz = np.fromfile('OUTPUT_FILES/Uz_file_single_v.bin', dtype=np.float32)
vz = vz.reshape(nrec, -1).T                  # (nt, nrec)
t = t0 + np.arange(vz.shape[0]) * dt
```

SPECFEM measures z upward from the bottom of the mesh and its force points
up (+z); receiver x in `STATIONS` includes the 400 m PML offset.
