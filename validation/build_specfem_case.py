"""
Generate SPECFEM2D inputs for the Marmousi model used by the FD solver.

Writes DATA/{Par_file, SOURCE, STATIONS, interfaces.dat} together with the
model in SPECFEM's tomography-file format, so that both codes see identical
material properties, source and receivers.

Two conventions have to be reconciled:

* z direction. The stored model has row 0 at the surface and z increasing
  downward; SPECFEM measures z upward from the bottom of the mesh, so the
  model is flipped when the tomography file is written.

* PML placement. SPECFEM puts the PML *inside* the declared domain, whereas
  the FD solver adds its padding *outside* the physical model. The model is
  therefore extended by the PML thickness on the sides and bottom (edge
  values repeated, matching how forward.py fills its padding), which keeps
  the full 10 km x 3 km physical domain clear of absorbing layers. The top is
  left untouched when a free surface is requested.

Run with: python build_specfem_case.py && ./run_this_example.sh

Author: Minh Nhat Tran
"""
import array
import os

import numpy as np

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(CASE_DIR), 'marmousi_models')
DATA_DIR = os.path.join(CASE_DIR, 'DATA')

DX = 20.0            # model grid spacing (m)
TMAX = 6.0           # simulation length (s)

# Source frequency. fc = 5 Hz gives ~3.5 FD grid points per shortest S
# wavelength at dx = 20 m (Vs_min = 881 m/s, fmax ~ 2.5 fc).
FC = 5.0

# Spectral-element size (m). With 4th-order GLL each element carries 4
# intervals, so 20 m at fc = 5 Hz yields ~14 GLL points per shortest
# S wavelength (SPECFEM suggests Ricker fc up to ~14 Hz).
ELEM_SIZE = 20.0
NELEM_PML = 20       # keeps the PML 400 m thick
PML = NELEM_PML * ELEM_SIZE

FREE_SURFACE = True  # False puts a PML on all four sides
SRC_DEPTH = 0.0      # source depth in metres below the physical top
REC_DEPTH = 0.0      # receiver depth in metres
SRC_X = 2000.0       # source position in metres
REC_STRIDE = 1       # receiver every REC_STRIDE model columns

NPROC = 16           # run with the system OpenMPI (see run_this_example.sh)


def write_record(fp, values):
    """Write one Fortran unformatted record: [int32 nbytes][data][int32 nbytes]."""
    data = array.array('f', np.asarray(values, dtype=np.float32).tolist())
    length = array.array('i', [data.itemsize * len(data)])
    length.tofile(fp)
    data.tofile(fp)
    length.tofile(fp)


def write_tomography(path, vp, vs, rho, dx):
    """Write the model in SPECFEM2D's binary tomography format.

    Input arrays use the FD convention (row 0 = surface). The file uses
    SPECFEM's convention: z upward, x varying fastest, first point at the
    bottom-left corner.
    """
    nz, nx = vp.shape
    end_x, end_z = (nx - 1) * dx, (nz - 1) * dx

    vp_f, vs_f, rho_f = vp[::-1], vs[::-1], rho[::-1]
    x = np.tile(np.arange(nx) * dx, nz)
    z = np.repeat(np.arange(nz) * dx, nx)
    records = np.column_stack([x, z, vp_f.ravel(), vs_f.ravel(), rho_f.ravel()])

    header = [0.0, 0.0, end_x, end_z, dx, dx, float(nx), float(nz),
              vp.min(), vp.max(), vs.min(), vs.max(), rho.min(), rho.max()]

    with open(path, 'wb') as fp:
        write_record(fp, header)
        write_record(fp, records.ravel())

    with open(path + '.info', 'w') as fp:
        fp.write("# tomography model generated from marmousi_models/*.npy\n")
        fp.write("# coordinate format   : x / z  # z-direction (positive up)\n")
        fp.write("#origin_x #origin_z #end_x #end_z\n")
        fp.write(f"0.0 0.0 {end_x:.1f} {end_z:.1f}\n")
        fp.write(f"#dx #dz\n{dx:.1f} {dx:.1f}\n")
        fp.write(f"#nx #nz\n{nx} {nz}\n")
        fp.write("#vp_min #vp_max #vs_min #vs_max #density_min #density_max\n")
        fp.write(f"{vp.min():f} {vp.max():f} {vs.min():f} {vs.max():f} "
                 f"{rho.min():f} {rho.max():f}\n")

    return end_x, end_z


def par_file(nelem_x, nelem_z, nstep, dt, nrec, lx, x_first, x_last, z_rec):
    absorb_top = '.false.' if FREE_SURFACE else '.true.'
    same_vertical = '.true.' if FREE_SURFACE else '.false.'
    return f"""#-----------------------------------------------------------
# simulation input parameters
#-----------------------------------------------------------
title                           = Marmousi - SPECFEM2D vs JAX SV-FDM

SIMULATION_TYPE                 = 1
NOISE_TOMOGRAPHY                = 0
SAVE_FORWARD                    = .false.

NPROC                           = {NPROC}
PARTITIONING_TYPE               = 3
NGNOD                           = 4

NSTEP                           = {nstep}
DT                              = {dt:.8e}
time_stepping_scheme            = 1

P_SV                            = .true.
AXISYM                          = .false.

setup_with_binary_database      = 0
MODEL                           = default
SAVE_MODEL                      = binary
TOMOGRAPHY_FILE                 = ./DATA/tomography_model_mine.xyz.bin

#-----------------------------------------------------------
# attenuation (disabled: the FD solver is purely elastic)
#-----------------------------------------------------------
ATTENUATION_VISCOELASTIC        = .false.
ATTENUATION_VISCOACOUSTIC       = .false.
N_SLS                           = 3
ATTENUATION_f0_REFERENCE        = {FC:.6f}
READ_VELOCITIES_AT_f0           = .false.
USE_SOLVOPT                     = .false.
ATTENUATION_PORO_FLUID_PART     = .false.
Q0_poroelastic                  = 1
freq0_poroelastic               = 10
ATTENUATION_PERMITTIVITY        = .false.
ATTENUATION_CONDUCTIVITY        = .false.
f0_electromagnetic              = 1.d9
UNDO_ATTENUATION_AND_OR_PML     = .false.
NT_DUMP_ATTENUATION             = 500
NO_BACKWARD_RECONSTRUCTION      = .false.

#-----------------------------------------------------------
# sources
#-----------------------------------------------------------
NSOURCES                        = 1
force_normal_to_surface         = .false.
initialfield                    = .false.
add_Bielak_conditions_bottom    = .false.
add_Bielak_conditions_right     = .false.
add_Bielak_conditions_top       = .false.
add_Bielak_conditions_left      = .false.
ACOUSTIC_FORCING                = .false.
noise_source_time_function_type = 4
write_moving_sources_database   = .false.

#-----------------------------------------------------------
# receivers
#-----------------------------------------------------------
seismotype                      = 2
NTSTEP_BETWEEN_OUTPUT_SEISMOS   = 100000
NTSTEP_BETWEEN_OUTPUT_SAMPLE    = 1
USE_TRICK_FOR_BETTER_PRESSURE   = .false.
USER_T0                         = 0.0d0
save_ASCII_seismograms          = .false.
save_binary_seismograms_single  = .true.
save_binary_seismograms_double  = .false.
SU_FORMAT                       = .false.

use_existing_STATIONS           = .true.
nreceiversets                   = 1
anglerec                        = 0.d0
rec_normal_to_surface           = .false.

nrec                            = {nrec}
xdeb                            = {x_first:.1f}
zdeb                            = {z_rec:.1f}
xfin                            = {x_last:.1f}
zfin                            = {z_rec:.1f}
record_at_surface_same_vertical = {same_vertical}

#-----------------------------------------------------------
# adjoint kernel outputs
#-----------------------------------------------------------
save_ASCII_kernels              = .true.
NTSTEP_BETWEEN_COMPUTE_KERNELS  = 1
APPROXIMATE_HESS_KL             = .false.

#-----------------------------------------------------------
# boundary conditions
#-----------------------------------------------------------
PML_BOUNDARY_CONDITIONS         = .true.
NELEM_PML_THICKNESS             = {NELEM_PML}
ROTATE_PML_ACTIVATE             = .false.
ROTATE_PML_ANGLE                = 30.
K_MIN_PML                       = 1.0d0
K_MAX_PML                       = 1.0d0
damping_change_factor_acoustic  = 0.5d0
damping_change_factor_elastic   = 1.0d0
PML_PARAMETER_ADJUSTMENT        = .false.

STACEY_ABSORBING_CONDITIONS     = .false.
ADD_PERIODIC_CONDITIONS         = .false.
PERIODIC_HORIZ_DIST             = 4000.d0

#-----------------------------------------------------------
# velocity and density model
#-----------------------------------------------------------
nbmodels                        = 1
1 -1 0.d0 0.d0 1.d0 0 0 0 0 0 0 0 0 0 0

interfacesfile                  = interfaces.dat

xmin                            = 0.d0
xmax                            = {lx:.1f}d0
nx                              = {nelem_x}

absorbbottom                    = .true.
absorbright                     = .true.
absorbtop                       = {absorb_top}
absorbleft                      = .true.

nbregions                       = 1
1 {nelem_x} 1 {nelem_z} 1

#-----------------------------------------------------------
# display parameters
#-----------------------------------------------------------
NTSTEP_BETWEEN_OUTPUT_INFO      = 500

output_grid_Gnuplot             = .false.
output_grid_ASCII               = .false.

OUTPUT_ENERGY                   = .false.
NTSTEP_BETWEEN_OUTPUT_ENERGY    = 10
COMPUTE_INTEGRATED_ENERGY_FIELD = .false.

NTSTEP_BETWEEN_OUTPUT_IMAGES    = 500
cutsnaps                        = 1.
output_color_image              = .false.
imagetype_JPEG                  = 5
factor_subsample_image          = 1.0d0
USE_CONSTANT_MAX_AMPLITUDE      = .false.
CONSTANT_MAX_AMPLITUDE_TO_USE   = 1.17d4
POWER_DISPLAY_COLOR             = 0.30d0
DRAW_SOURCES_AND_RECEIVERS      = .true.
DRAW_WATER_IN_BLUE              = .false.
USE_SNAPSHOT_NUMBER_IN_FILENAME = .false.

output_postscript_snapshot      = .false.
imagetype_postscript            = 1
meshvect                        = .true.
modelvect                       = .false.
boundvect                       = .true.
interpol                        = .true.
pointsdisp                      = 6
subsamp_postscript              = 1
sizemax_arrows                  = 1.d0
US_LETTER                       = .false.

output_wavefield_dumps          = .false.
imagetype_wavefield_dumps       = 1
use_binary_for_wavefield_dumps  = .false.

#-----------------------------------------------------------
# mesh
#-----------------------------------------------------------
read_external_mesh              = .false.
mesh_file                       = ./DATA/mesh_file
nodes_coords_file               = ./DATA/nodes_coords_file
materials_file                  = ./DATA/materials_file
free_surface_file               = ./DATA/free_surface_file
axial_elements_file             = ./DATA/axial_elements_file
absorbing_surface_file          = ./DATA/absorbing_surface_file
acoustic_forcing_surface_file   = ./DATA/MSH/Surf_acforcing_Bottom_enforcing_mesh
absorbing_cpml_file             = ./DATA/absorbing_cpml_file
tangential_detection_curve_file = ./DATA/courbe_eros_nodes

NUMBER_OF_SIMULTANEOUS_RUNS     = 1
BROADCAST_SAME_MESH_AND_MODEL   = .true.

GPU_MODE                        = .false.
"""


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(CASE_DIR, 'OUTPUT_FILES'), exist_ok=True)

    vp = np.load(os.path.join(MODEL_DIR, 'vp_true.npy')).astype(np.float64)
    vs = np.load(os.path.join(MODEL_DIR, 'vs_true.npy')).astype(np.float64)
    rho = np.load(os.path.join(MODEL_DIR, 'rho_true.npy')).astype(np.float64)
    nz0, nx0 = vs.shape
    print(f"model {nz0} x {nx0} = {(nx0 - 1) * DX:.0f} x {(nz0 - 1) * DX:.0f} m")

    # extend the model so that the PML falls outside the physical domain
    npad = int(round(PML / DX))
    ntop = 0 if FREE_SURFACE else npad
    vp, vs, rho = (np.pad(a, ((ntop, npad), (npad, npad)), mode='edge')
                   for a in (vp, vs, rho))
    nz, nx = vs.shape
    sides = 'sides and bottom' if FREE_SURFACE else 'all four sides'
    print(f"  + {PML:.0f} m PML on {sides} -> {nz} x {nx}")

    lx, lz = write_tomography(
        os.path.join(DATA_DIR, 'tomography_model_mine.xyz.bin'),
        vp, vs, rho, DX)
    print(f"  mesh domain {lx:.0f} x {lz:.0f} m")

    nelem_x = int(round(lx / ELEM_SIZE))
    nelem_z = int(round(lz / ELEM_SIZE))
    # CFL for 4th-order GLL: the closest point pair inside an element is
    # about 0.17 of its size
    dt = 0.3 * (ELEM_SIZE * 0.17) / vp.max()
    nstep = int(round(TMAX / dt))
    print(f"  mesh {nelem_x} x {nelem_z} = {nelem_x * nelem_z} elements, "
          f"dt = {dt * 1e3:.4f} ms, nstep = {nstep}")

    with open(os.path.join(DATA_DIR, 'interfaces.dat'), 'w') as f:
        f.write(f"# number of interfaces\n2\n#\n"
                f"# interface 1 (bottom of the mesh)\n 2\n"
                f"     0.d0     0.d0\n{lx:10.1f}d0     0.d0\n"
                f"# interface 2 (top surface)\n 2\n"
                f"     0.d0{lz:10.1f}d0\n{lx:10.1f}d0{lz:10.1f}d0\n#\n"
                f"# number of spectral elements in the vertical direction\n"
                f" {nelem_z}\n")

    # coordinates in the mesh frame: physical x plus the left padding,
    # and z measured upward from the bottom
    z_top = lz - ntop * DX
    src_x = SRC_X + PML
    src_z = z_top - SRC_DEPTH
    rec_z = z_top - REC_DEPTH

    with open(os.path.join(DATA_DIR, 'SOURCE'), 'w') as f:
        f.write(f"""source_surf                     = .false.
xs                              = {src_x:.1f}
zs                              = {src_z:.1f}
source_type                     = 1
time_function_type              = 1
name_of_source_file             = YYYYYYYYYYYYYYYYYY
burst_band_width                = 0.
f0                              = {FC:.1f}
tshift                          = 0.0
anglesource                     = 0.
Mxx                             = 1.
Mzz                             = 1.
Mxz                             = 0.
factor                          = 1.d0
vx                              = 0.0
vz                              = 0.0
""")

    rec_ix = np.arange(0, nx0, REC_STRIDE)
    with open(os.path.join(DATA_DIR, 'STATIONS'), 'w') as f:
        for i, ix in enumerate(rec_ix):
            f.write(f"S{i + 1:04d}    AA {ix * DX + PML:20.7f} "
                    f"{rec_z:20.7f}       0.0         0.0\n")

    with open(os.path.join(DATA_DIR, 'Par_file'), 'w') as f:
        f.write(par_file(nelem_x, nelem_z, nstep, dt, len(rec_ix), lx,
                         PML, rec_ix[-1] * DX + PML, rec_z))

    print(f"  vertical force at x = {SRC_X:.0f} m, depth {SRC_DEPTH:.0f} m, "
          f"f0 = {FC:g} Hz")
    print(f"  {len(rec_ix)} receivers at depth {REC_DEPTH:.0f} m, "
          f"free surface = {FREE_SURFACE}")
    print(f"\nWritten to {DATA_DIR}/")


if __name__ == '__main__':
    main()
