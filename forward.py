"""
fwi/forward.py — 2D P-SV Elastic Wave Forward Modeling (JAX)

Velocity-STRAIN Finite Difference with C-PML (2nd/4th/8th Order)
Combines: wave propagation, Ricker wavelet, and PML absorbing boundaries.
Force type: Only vertical

Public API:
    ricker_jax(t, fc, t0)          — generate source wavelet
    build_forward_fn(...)          — returns JIT-compiled forward function
    forward_jax(...)               — raw forward modeling (used internally)

Authors: Minh Nhat Tran
Date: February 2026
"""
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# Ricker wavelet
# ─────────────────────────────────────────────────────────────────────────────

def ricker_jax(t, fc, t0, scaling_factor=1):
    """Generate Ricker wavelet using JAX."""
    arg = (jnp.pi * fc * (t - t0)) ** 2
    return scaling_factor * (1.0 - 2.0 * arg) * jnp.exp(-arg)


# ─────────────────────────────────────────────────────────────────────────────
# DAS gauge-length averaging
# ─────────────────────────────────────────────────────────────────────────────

def _gauge_taps(gauge_length, dx):
    """Number of grid POINTS touched by a DAS cable of length L.

    L is the distance between the two cable ends, so the cable spans
    round(L / dx) grid intervals → round(L / dx) + 1 nodes:
        gauge_length=20, dx=20 → 1 interval → 2 points  (eps[i]+eps[i+1])/2
        gauge_length=40, dx=20 → 2 intervals → 3 points
    """
    return int(round(gauge_length / dx)) + 1


def _gauge_length_average_trace(trace, n_taps):
    """DAS gauge-length averaging of a strain TRACE along x (1-D over channels).

    Operates on the strain already sampled over the PHYSICAL domain (one value per
    domain column, shape (nx_dom,)) — NOT on the padded simulation grid — so the
    PML never enters the average. DAS integrates strain over a cable of length
    L = n_taps * dx with a FORWARD moving window sliding by ONE node (overlap):
        out[i] = (eps[i] + eps[i+1] + ... + eps[i+n_taps-1]) / n_taps
    Examples (n_taps=2, L=20 m, dx=20 m):
        out[1] = (eps[1] + eps[2]) / 2
        out[2] = (eps[2] + eps[3]) / 2   ← slides by one node, overlaps out[1]

    EDGE HANDLING — a real DAS channel needs the FULL gauge length. A node whose
    forward window would run past the right edge (i + n_taps > N) does NOT have
    enough points to form a true gauge-length average, so it is set to ZERO
    rather than shrinking the window. The output keeps the SAME N columns; the
    last (n_taps - 1) are zero. Observed and predicted are zeroed at the same
    columns, so the residual there is exactly zero — harmless to the misfit.

    Implemented with a cumulative-sum sliding window (not jnp.convolve): the
    GPU XLA convolution path produced NaN gradients on the tiny (~1e-13) strain
    field inside the checkpointed time-scan. cumsum uses only add/sub/slice —
    finite-safe and differentiable on every backend.
    """
    if n_taps <= 1:
        return trace
    (nx,) = trace.shape
    prefix = jnp.concatenate([jnp.zeros((1,), trace.dtype), jnp.cumsum(trace)])
    starts = jnp.arange(nx)
    ends = starts + n_taps
    valid = ends <= nx
    ends_clamped = jnp.minimum(ends, nx)
    out = (prefix[ends_clamped] - prefix[starts]) / n_taps
    out = jnp.where(valid, out, 0.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# C-PML (Convolutional Perfectly Matched Layer)
# ─────────────────────────────────────────────────────────────────────────────

def _pml_coefficients(pad, dx, dt, Vp_max, fc, nz_t, nx_t, nz_dom,
                      free_surface=True):
    """
    Setup C-PML coefficients using JAX.
    Returns JAX arrays for use in JIT-compiled functions.

    When free_surface=True:  PML on left, right, bottom only (3 sides).
    When free_surface=False: PML on all 4 sides including top.
    """
    N = 2              # Polynomial order
    Rc = 1e-3          # Reflection coefficient
    L_pml = pad * dx   # PML thickness

    d0 = -(N + 1) * Vp_max * jnp.log(Rc) / (2 * L_pml)

    i_arr = jnp.arange(pad)
    dist = (i_arr + 1) * dx / L_pml
    d_profile = d0 * (dist ** N)
    k_profile = 1.0 + (2.0 - 1.0) * (dist ** N)
    alpha_profile = jnp.pi * fc * (1.0 - dist)

    dx_arr = jnp.zeros((nz_t, nx_t))
    dz_arr = jnp.zeros((nz_t, nx_t))
    kx = jnp.ones((nz_t, nx_t))
    kz = jnp.ones((nz_t, nx_t))
    alpha_x = jnp.zeros((nz_t, nx_t))
    alpha_z = jnp.zeros((nz_t, nx_t))

    # Left PML
    for i in range(pad):
        col = pad - 1 - i
        dx_arr = dx_arr.at[:, col].set(d_profile[i])
        kx = kx.at[:, col].set(k_profile[i])
        alpha_x = alpha_x.at[:, col].set(alpha_profile[i])

    # Right PML
    for i in range(pad):
        col = nx_t - pad + i
        dx_arr = dx_arr.at[:, col].set(d_profile[i])
        kx = kx.at[:, col].set(k_profile[i])
        alpha_x = alpha_x.at[:, col].set(alpha_profile[i])

    # Bottom PML
    bottom_start = nz_dom if free_surface else pad + nz_dom
    for i in range(pad):
        row = bottom_start + i
        dz_arr = dz_arr.at[row, :].set(d_profile[i])
        kz = kz.at[row, :].set(k_profile[i])
        alpha_z = alpha_z.at[row, :].set(alpha_profile[i])

    # Top PML (only when no free surface)
    if not free_surface:
        for i in range(pad):
            row = pad - 1 - i
            dz_arr = dz_arr.at[row, :].set(d_profile[i])
            kz = kz.at[row, :].set(k_profile[i])
            alpha_z = alpha_z.at[row, :].set(alpha_profile[i])

    bx = jnp.exp(-(dx_arr / kx + alpha_x) * dt)
    bz = jnp.exp(-(dz_arr / kz + alpha_z) * dt)

    denom_x = kx * (dx_arr + kx * alpha_x)
    denom_z = kz * (dz_arr + kz * alpha_z)

    ax = jnp.where(denom_x > 1e-10, dx_arr * (bx - 1.0) / denom_x, 0.0)
    az = jnp.where(denom_z > 1e-10, dz_arr * (bz - 1.0) / denom_z, 0.0)

    return kx, kz, bx, bz, ax, az


@jax.jit
def _pml_apply(D, k, psi, b, a):
    """Apply C-PML to derivative (JIT-compiled)"""
    psi_new = b * psi + a * D
    D_new = D / k + psi_new
    return D_new, psi_new


# ─────────────────────────────────────────────────────────────────────────────
# Forward modeling (core)
# ─────────────────────────────────────────────────────────────────────────────

def forward_jax(Vs, Vp, rho, src_x, src_z, rec_x, rec_z,
                nx_dom, nz_dom, dx, dz, dt, nt, fc, pad,
                src_wavelet,
                return_wavefields=False, block_size=100, return_vars=None,
                fd_order=2, free_surface=True,
                gauge_length=None, component=None):
    """
    2D P-SV Elastic Wave Forward Modeling - JAX Version (Velocity-Strain Formulation)
    2nd/4th/8th Order Spatial Accuracy with Block Checkpointing

    Parameters:
    -----------
    Vs, Vp, rho : numpy arrays
        Velocity and density models (nz_dom, nx_dom)
    src_x, src_z : int
        Source position indices
    rec_x, rec_z : numpy arrays
        Receiver position indices
    nx_dom, nz_dom : int
        Domain size
    dx, dz, dt : float
        Grid spacing and time step
    nt : int
        Number of time steps
    fc : float
        Source frequency (used for PML tuning)
    pad : int
        PML padding size
    src_wavelet : jax array
        Source wavelet (nt,) — must be created externally (e.g. via ricker_jax)
    return_wavefields : bool
        Whether to return the full wavefields or just receiver data.
    block_size : int
        Number of time steps per checkpoint block.
    return_vars : list of str, optional
        Specific variables to return. If provided, overrides return_wavefields.
        Options: 'vx', 'vz', 'ex', 'ez' (receiver data);
                 'vx_full', 'vz_full', 'ex_full', 'ez_full', 'es_full' (full wavefields).
    fd_order : int
        Finite difference spatial order (2, 4, or 8). Default: 2.
    free_surface : bool
        If True, apply free surface BC at top (z=0). PML on 3 sides.
        If False, apply PML on all 4 sides including top. Default: True.

    Returns:
    --------
    results : tuple
        Requested variables as JAX arrays. Default: (rec_vx, rec_vz, rec_exx, rec_ezz)
    """
    Vs = jnp.array(Vs)
    Vp = jnp.array(Vp)
    rho = jnp.array(rho)
    rec_x = jnp.array(rec_x)
    rec_z = jnp.array(rec_z)

    if fd_order == 8:
        c1, c2, c3, c4 = 1225.0/1024.0, -245.0/3072.0, 49.0/5120.0, -5.0/7168.0
    elif fd_order >= 4:
        c1, c2, c3, c4 = 9.0/8.0, -1.0/24.0, 0.0, 0.0
    else:
        c1, c2, c3, c4 = 1.0, 0.0, 0.0, 0.0

    # Grid size
    if free_surface:
        nx_t, nz_t = nx_dom + 2 * pad, nz_dom + pad
        z_start = 0       # domain starts at row 0
    else:
        nx_t, nz_t = nx_dom + 2 * pad, nz_dom + 2 * pad
        z_start = pad      # domain starts at row pad
    nz, nx = nz_t, nx_t

    Vs_full = jnp.zeros((nz, nx)).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(Vs)
    Vp_full = jnp.zeros((nz, nx)).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(Vp)
    rho_full = jnp.zeros((nz, nx)).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(rho)

    # Extend to left PML region
    Vs_full = Vs_full.at[z_start:z_start+nz_dom, :pad].set(Vs_full[z_start:z_start+nz_dom, pad:pad+1])
    Vp_full = Vp_full.at[z_start:z_start+nz_dom, :pad].set(Vp_full[z_start:z_start+nz_dom, pad:pad+1])
    rho_full = rho_full.at[z_start:z_start+nz_dom, :pad].set(rho_full[z_start:z_start+nz_dom, pad:pad+1])

    # Extend to right PML region
    Vs_full = Vs_full.at[z_start:z_start+nz_dom, nx_t-pad:].set(Vs_full[z_start:z_start+nz_dom, pad+nx_dom-1:pad+nx_dom])
    Vp_full = Vp_full.at[z_start:z_start+nz_dom, nx_t-pad:].set(Vp_full[z_start:z_start+nz_dom, pad+nx_dom-1:pad+nx_dom])
    rho_full = rho_full.at[z_start:z_start+nz_dom, nx_t-pad:].set(rho_full[z_start:z_start+nz_dom, pad+nx_dom-1:pad+nx_dom])

    # Extend to bottom PML region
    Vs_full = Vs_full.at[z_start+nz_dom:, :].set(Vs_full[z_start+nz_dom-1:z_start+nz_dom, :])
    Vp_full = Vp_full.at[z_start+nz_dom:, :].set(Vp_full[z_start+nz_dom-1:z_start+nz_dom, :])
    rho_full = rho_full.at[z_start+nz_dom:, :].set(rho_full[z_start+nz_dom-1:z_start+nz_dom, :])

    # Extend to top PML region (only when no free surface)
    if not free_surface:
        Vs_full = Vs_full.at[:pad, :].set(Vs_full[pad:pad+1, :])
        Vp_full = Vp_full.at[:pad, :].set(Vp_full[pad:pad+1, :])
        rho_full = rho_full.at[:pad, :].set(rho_full[pad:pad+1, :])

    Vs_full  = lax.stop_gradient(Vs_full).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(Vs)
    Vp_full  = lax.stop_gradient(Vp_full).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(Vp)
    rho_full = lax.stop_gradient(rho_full).at[z_start:z_start+nz_dom, pad:pad+nx_dom].set(rho)

    # Material parameters
    M = rho_full * Vs_full**2        # μ
    L2M = rho_full * Vp_full**2      # λ + 2μ
    L = L2M - 2 * M                  # λ

    # Buoyancy (inverse density)
    B = 1.0 / rho_full

    # Free surface coefficient
    if free_surface:
        L_ratio = -L / jnp.maximum(L2M, 1e-12)

    # PML coefficients
    Vp_max = jnp.max(Vp)
    Vp_max = lax.stop_gradient(Vp_max)
    kx, kz, bx, bz, ax, az = _pml_coefficients(
        pad, dx, dt, Vp_max, fc, nz, nx, nz_dom, free_surface=free_surface)

    # Source positions
    sx, sz = src_x + pad, src_z + z_start

    # DAS gauge length. component map: 0=vx, 1=vz, 2=exx, 3=ezz.
    gauge_taps = _gauge_taps(gauge_length, dx) if gauge_length else 1
    apply_gauge = gauge_taps > 1

    dom_cols = jnp.arange(nx_dom, dtype=jnp.int32) + pad
    dom_rz = jnp.broadcast_to((rec_z[0] + z_start).astype(jnp.int32), (nx_dom,))

    initial_carry = (jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)),
                     jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)),
                     jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)), jnp.zeros((nz, nx)))

    # Define time_step
    def time_step(carry, it):
        U, V, ex, ez, es, p_Dx_L2M_ex, p_Dx_L_ez, p_Dz_M_es, p_Dz_L2M_ez, p_Dz_L_ex, p_Dx_M_es, p_Dx_U, p_Dz_V, p_Dz_U, p_Dx_V = carry

        # ── 1) Update U (vx) ──────────────────────────────────────────────
        L2M_ex, L_ez, M_es = L2M * ex, L * ez, 2 * M * es

        Dx_L2M_ex = jnp.zeros((nz, nx))
        Dx_L2M_ex = Dx_L2M_ex.at[:nz-1, :nx-1].set(
            c1 * (L2M_ex[:nz-1, 1:nx] - L2M_ex[:nz-1, :nx-1]))
        if fd_order >= 4:
            Dx_L2M_ex = Dx_L2M_ex.at[:nz-1, 1:nx-2].add(
                c2 * (L2M_ex[:nz-1, 3:nx] - L2M_ex[:nz-1, :nx-3]))
        if fd_order >= 8:
            Dx_L2M_ex = Dx_L2M_ex.at[:nz-1, 2:nx-3].add(
                c3 * (L2M_ex[:nz-1, 5:nx] - L2M_ex[:nz-1, :nx-5]))
            Dx_L2M_ex = Dx_L2M_ex.at[:nz-1, 3:nx-4].add(
                c4 * (L2M_ex[:nz-1, 7:nx] - L2M_ex[:nz-1, :nx-7]))
        Dx_L2M_ex, p_Dx_L2M_ex = _pml_apply(Dx_L2M_ex, kx, p_Dx_L2M_ex, bx, ax)

        Dx_L_ez = jnp.zeros((nz, nx))
        Dx_L_ez = Dx_L_ez.at[:nz-1, :nx-1].set(
            c1 * (L_ez[:nz-1, 1:nx] - L_ez[:nz-1, :nx-1]))
        if fd_order >= 4:
            Dx_L_ez = Dx_L_ez.at[:nz-1, 1:nx-2].add(
                c2 * (L_ez[:nz-1, 3:nx] - L_ez[:nz-1, :nx-3]))
        if fd_order >= 8:
            Dx_L_ez = Dx_L_ez.at[:nz-1, 2:nx-3].add(
                c3 * (L_ez[:nz-1, 5:nx] - L_ez[:nz-1, :nx-5]))
            Dx_L_ez = Dx_L_ez.at[:nz-1, 3:nx-4].add(
                c4 * (L_ez[:nz-1, 7:nx] - L_ez[:nz-1, :nx-7]))
        Dx_L_ez, p_Dx_L_ez = _pml_apply(Dx_L_ez, kx, p_Dx_L_ez, bx, ax)

        Dz_M_es = jnp.zeros((nz, nx))
        Dz_M_es = Dz_M_es.at[1:nz-1, 1:nx-1].set(
            c1 * (M_es[1:nz-1, 1:nx-1] - M_es[:nz-2, 1:nx-1]))
        if fd_order >= 4:
            Dz_M_es = Dz_M_es.at[2:nz-1, 1:nx-1].add(
                c2 * (M_es[3:nz, 1:nx-1] - M_es[:nz-3, 1:nx-1]))
        if fd_order >= 8:
            Dz_M_es = Dz_M_es.at[3:nz-2, 1:nx-1].add(
                c3 * (M_es[5:nz, 1:nx-1] - M_es[:nz-5, 1:nx-1]))
            Dz_M_es = Dz_M_es.at[4:nz-3, 1:nx-1].add(
                c4 * (M_es[7:nz, 1:nx-1] - M_es[:nz-7, 1:nx-1]))
        if free_surface:
            f = M_es[:, 1:nx-1]
            Dz_M_es = Dz_M_es.at[0, 1:nx-1].set(
                2*c1*f[0] + 2*c2*f[1] + 2*c3*f[2] + 2*c4*f[3])
            if fd_order >= 4:
                Dz_M_es = Dz_M_es.at[1, 1:nx-1].add(c2 * (f[0] + f[2]))
            if fd_order >= 8:
                Dz_M_es = Dz_M_es.at[1, 1:nx-1].add(c3 * (f[1] + f[3]) + c4 * (f[2] + f[4]))
                Dz_M_es = Dz_M_es.at[2, 1:nx-1].add(c3 * (f[0] + f[4]) + c4 * (f[1] + f[5]))
                Dz_M_es = Dz_M_es.at[3, 1:nx-1].add(c4 * (f[0] + f[6]))
        Dz_M_es, p_Dz_M_es = _pml_apply(Dz_M_es, kz, p_Dz_M_es, bz, az)

        U = U.at[0:nz-1, 1:nx-1].set(
            U[0:nz-1, 1:nx-1]
            + dt/dx * B[0:nz-1, 1:nx-1] * Dx_L2M_ex[0:nz-1, 1:nx-1]
            + dt/dx * B[0:nz-1, 1:nx-1] * Dx_L_ez[0:nz-1, 1:nx-1]
            + dt/dz * B[0:nz-1, 1:nx-1] * Dz_M_es[0:nz-1, 1:nx-1]
        )

        # ── 2) Update V (vz) ──────────────────────────────────────────────
        L2M_ez, L_ex = L2M * ez, L * ex

        Dx_M_es = jnp.zeros((nz, nx))
        Dx_M_es = Dx_M_es.at[:nz-1, 1:nx-1].set(
            c1 * (M_es[:nz-1, 1:nx-1] - M_es[:nz-1, :nx-2]))
        if fd_order >= 4:
            Dx_M_es = Dx_M_es.at[:nz-1, 2:nx-1].add(
                c2 * (M_es[:nz-1, 3:nx] - M_es[:nz-1, :nx-3]))
        if fd_order >= 8:
            Dx_M_es = Dx_M_es.at[:nz-1, 3:nx-2].add(
                c3 * (M_es[:nz-1, 5:nx] - M_es[:nz-1, :nx-5]))
            Dx_M_es = Dx_M_es.at[:nz-1, 4:nx-3].add(
                c4 * (M_es[:nz-1, 7:nx] - M_es[:nz-1, :nx-7]))
        Dx_M_es, p_Dx_M_es = _pml_apply(Dx_M_es, kx, p_Dx_M_es, bx, ax)

        Dz_L2M_ez = jnp.zeros((nz, nx))
        Dz_L2M_ez = Dz_L2M_ez.at[:nz-1, 1:nx-1].set(
            c1 * (L2M_ez[1:nz, 1:nx-1] - L2M_ez[:nz-1, 1:nx-1]))
        if fd_order >= 4:
            Dz_L2M_ez = Dz_L2M_ez.at[1:nz-2, 1:nx-1].add(
                c2 * (L2M_ez[3:nz, 1:nx-1] - L2M_ez[:nz-3, 1:nx-1]))
        if fd_order >= 8:
            Dz_L2M_ez = Dz_L2M_ez.at[2:nz-3, 1:nx-1].add(
                c3 * (L2M_ez[5:nz, 1:nx-1] - L2M_ez[:nz-5, 1:nx-1]))
            Dz_L2M_ez = Dz_L2M_ez.at[3:nz-4, 1:nx-1].add(
                c4 * (L2M_ez[7:nz, 1:nx-1] - L2M_ez[:nz-7, 1:nx-1]))
        if free_surface:
            if fd_order >= 4:
                Dz_L2M_ez = Dz_L2M_ez.at[0, 1:nx-1].add(
                    c2 * (L2M_ez[2, 1:nx-1]
                          + L2M_ez[1, 1:nx-1] + L_ex[1, 1:nx-1] + L_ex[0, 1:nx-1]))
            if fd_order >= 8:
                Dz_L2M_ez = Dz_L2M_ez.at[0, 1:nx-1].add(
                    c3 * (L2M_ez[3, 1:nx-1]
                          + L2M_ez[2, 1:nx-1] + L_ex[2, 1:nx-1] + L_ex[1, 1:nx-1]))
                Dz_L2M_ez = Dz_L2M_ez.at[0, 1:nx-1].add(
                    c4 * (L2M_ez[4, 1:nx-1]
                          + L2M_ez[3, 1:nx-1] + L_ex[3, 1:nx-1] + L_ex[2, 1:nx-1]))
                Dz_L2M_ez = Dz_L2M_ez.at[1, 1:nx-1].add(
                    c3 * (L2M_ez[4, 1:nx-1]
                          + L2M_ez[1, 1:nx-1] + L_ex[1, 1:nx-1] + L_ex[0, 1:nx-1]))
                Dz_L2M_ez = Dz_L2M_ez.at[1, 1:nx-1].add(
                    c4 * (L2M_ez[5, 1:nx-1]
                          + L2M_ez[2, 1:nx-1] + L_ex[2, 1:nx-1] + L_ex[1, 1:nx-1]))
                Dz_L2M_ez = Dz_L2M_ez.at[2, 1:nx-1].add(
                    c4 * (L2M_ez[6, 1:nx-1]
                          + L2M_ez[1, 1:nx-1] + L_ex[1, 1:nx-1] + L_ex[0, 1:nx-1]))
        Dz_L2M_ez, p_Dz_L2M_ez = _pml_apply(Dz_L2M_ez, kz, p_Dz_L2M_ez, bz, az)

        Dz_L_ex = jnp.zeros((nz, nx))
        Dz_L_ex = Dz_L_ex.at[:nz-1, 1:nx-1].set(
            c1 * (L_ex[1:nz, 1:nx-1] - L_ex[:nz-1, 1:nx-1]))
        if fd_order >= 4:
            Dz_L_ex = Dz_L_ex.at[1:nz-2, 1:nx-1].add(
                c2 * (L_ex[3:nz, 1:nx-1] - L_ex[:nz-3, 1:nx-1]))
        if fd_order >= 8:
            Dz_L_ex = Dz_L_ex.at[2:nz-3, 1:nx-1].add(
                c3 * (L_ex[5:nz, 1:nx-1] - L_ex[:nz-5, 1:nx-1]))
            Dz_L_ex = Dz_L_ex.at[3:nz-4, 1:nx-1].add(
                c4 * (L_ex[7:nz, 1:nx-1] - L_ex[:nz-7, 1:nx-1]))
        if free_surface:
            if fd_order >= 4:
                Dz_L_ex = Dz_L_ex.at[0, 1:nx-1].add(
                    c2 * (L_ex[2, 1:nx-1] - L_ex[0, 1:nx-1]))
            if fd_order >= 8:
                Dz_L_ex = Dz_L_ex.at[0, 1:nx-1].add(
                    c3 * (L_ex[3, 1:nx-1] - L_ex[1, 1:nx-1])
                    + c4 * (L_ex[4, 1:nx-1] - L_ex[2, 1:nx-1]))
                Dz_L_ex = Dz_L_ex.at[1, 1:nx-1].add(
                    c3 * (L_ex[4, 1:nx-1] - L_ex[0, 1:nx-1])
                    + c4 * (L_ex[5, 1:nx-1] - L_ex[1, 1:nx-1]))
                Dz_L_ex = Dz_L_ex.at[2, 1:nx-1].add(
                    c4 * (L_ex[6, 1:nx-1] - L_ex[0, 1:nx-1]))
        Dz_L_ex, p_Dz_L_ex = _pml_apply(Dz_L_ex, kz, p_Dz_L_ex, bz, az)

        V = V.at[0:nz-1, 1:nx-1].set(
            V[0:nz-1, 1:nx-1]
            + dt/dx * B[0:nz-1, 1:nx-1] * Dx_M_es[0:nz-1, 1:nx-1]
            + dt/dz * B[0:nz-1, 1:nx-1] * Dz_L2M_ez[0:nz-1, 1:nx-1]
            + dt/dz * B[0:nz-1, 1:nx-1] * Dz_L_ex[0:nz-1, 1:nx-1]
        )

        # Force source injection
        V = V.at[sz, sx].add(dt/(dx * dz) * B[sz, sx] * src_wavelet[it])

        # ── 3) Update εxx ─────────────────────────────────────────────────
        Dx_U = jnp.zeros((nz, nx))
        Dx_U = Dx_U.at[:nz, 1:nx].set(
            c1 * (U[:nz, 1:nx] - U[:nz, :nx-1]))
        if fd_order >= 4:
            Dx_U = Dx_U.at[:nz, 2:nx-1].add(
                c2 * (U[:nz, 3:nx] - U[:nz, :nx-3]))
        if fd_order >= 8:
            Dx_U = Dx_U.at[:nz, 3:nx-2].add(
                c3 * (U[:nz, 5:nx] - U[:nz, :nx-5]))
            Dx_U = Dx_U.at[:nz, 4:nx-3].add(
                c4 * (U[:nz, 7:nx] - U[:nz, :nx-7]))
        Dx_U, p_Dx_U = _pml_apply(Dx_U, kx, p_Dx_U, bx, ax)
        ex = ex.at[:, 1:nx].add(dt/dx * Dx_U[:, 1:nx])

        # ── 4) Update εzz ─────────────────────────────────────────────────
        Dz_V = jnp.zeros((nz, nx))
        if free_surface:
            Dz_V = Dz_V.at[0, :nx-1].set(L_ratio[0, :nx-1] * Dx_U[0, :nx-1] * dz / dx)
        Dz_V = Dz_V.at[1:nz, :nx-1].set(
            c1 * (V[1:nz, :nx-1] - V[0:nz-1, :nx-1]))
        if fd_order >= 4:
            Dz_V = Dz_V.at[2:nz-1, :nx-1].add(
                c2 * (V[3:nz, :nx-1] - V[:nz-3, :nx-1]))
        if fd_order >= 8:
            Dz_V = Dz_V.at[3:nz-2, :nx-1].add(
                c3 * (V[5:nz, :nx-1] - V[:nz-5, :nx-1]))
            Dz_V = Dz_V.at[4:nz-3, :nx-1].add(
                c4 * (V[7:nz, :nx-1] - V[:nz-7, :nx-1]))
        if free_surface:
            if fd_order >= 4:
                Dz_V = Dz_V.at[1, :nx-1].add(
                    c2 * (V[0, :nx-1] + V[2, :nx-1]))
            if fd_order >= 8:
                Dz_V = Dz_V.at[1, :nx-1].add(
                    c3 * (V[1, :nx-1] + V[3, :nx-1])
                    + c4 * (V[2, :nx-1] + V[4, :nx-1]))
                Dz_V = Dz_V.at[2, :nx-1].add(
                    c3 * (V[0, :nx-1] + V[4, :nx-1])
                    + c4 * (V[1, :nx-1] + V[5, :nx-1]))
                Dz_V = Dz_V.at[3, :nx-1].add(
                    c4 * (V[0, :nx-1] + V[6, :nx-1]))
        Dz_V, p_Dz_V = _pml_apply(Dz_V, kz, p_Dz_V, bz, az)
        ez = ez.at[0:nz, :].add(dt/dz * Dz_V[0:nz, :])

        # ── 5) Update εxz ─────────────────────────────────────────────────
        Dz_U = jnp.zeros((nz, nx))
        Dz_U = Dz_U.at[:nz-1, 1:nx].set(
            c1 * (U[1:nz, 1:nx] - U[:nz-1, 1:nx]))
        if fd_order >= 4:
            Dz_U = Dz_U.at[1:nz-2, 1:nx].add(
                c2 * (U[3:nz, 1:nx] - U[:nz-3, 1:nx]))
        if fd_order >= 8:
            Dz_U = Dz_U.at[2:nz-3, 1:nx].add(
                c3 * (U[5:nz, 1:nx] - U[:nz-5, 1:nx]))
            Dz_U = Dz_U.at[3:nz-4, 1:nx].add(
                c4 * (U[7:nz, 1:nx] - U[:nz-7, 1:nx]))
        if free_surface:
            if fd_order >= 4:
                Dz_U = Dz_U.at[0, 1:nx].add(
                    c2 * (U[2, 1:nx] - U[0, 1:nx]))
            if fd_order >= 8:
                Dz_U = Dz_U.at[0, 1:nx].add(
                    c3 * (U[3, 1:nx] - U[1, 1:nx])
                    + c4 * (U[4, 1:nx] - U[2, 1:nx]))
                Dz_U = Dz_U.at[1, 1:nx].add(
                    c3 * (U[4, 1:nx] - U[0, 1:nx])
                    + c4 * (U[5, 1:nx] - U[1, 1:nx]))
                Dz_U = Dz_U.at[2, 1:nx].add(
                    c4 * (U[6, 1:nx] - U[0, 1:nx]))
        Dz_U, p_Dz_U = _pml_apply(Dz_U, kz, p_Dz_U, bz, az)

        Dx_V = jnp.zeros((nz, nx))
        Dx_V = Dx_V.at[:nz-1, :nx-1].set(
            c1 * (V[:nz-1, 1:nx] - V[:nz-1, :nx-1]))
        if fd_order >= 4:
            Dx_V = Dx_V.at[:nz-1, 1:nx-2].add(
                c2 * (V[:nz-1, 3:nx] - V[:nz-1, :nx-3]))
        if fd_order >= 8:
            Dx_V = Dx_V.at[:nz-1, 2:nx-3].add(
                c3 * (V[:nz-1, 5:nx] - V[:nz-1, :nx-5]))
            Dx_V = Dx_V.at[:nz-1, 3:nx-4].add(
                c4 * (V[:nz-1, 7:nx] - V[:nz-1, :nx-7]))
        Dx_V, p_Dx_V = _pml_apply(Dx_V, kx, p_Dx_V, bx, ax)

        es = es.at[:nz-1, :nx-1].add(0.5 * (dt/dz * Dz_U[:nz-1, :nx-1] + dt/dx * Dx_V[:nz-1, :nx-1]))

        new_carry = (U, V, ex, ez, es, p_Dx_L2M_ex, p_Dx_L_ez, p_Dz_M_es,
                     p_Dz_L2M_ez, p_Dz_L_ex, p_Dx_M_es, p_Dx_U, p_Dz_V, p_Dz_U, p_Dx_V)

        def _measure(fld, comp_idx):
            dom = fld[dom_rz, dom_cols]
            if apply_gauge and comp_idx == component:
                return _gauge_length_average_trace(dom, gauge_taps)
            return dom[rec_x]

        vx_o = _measure(U, 0)
        vz_o = _measure(V, 1)
        ex_o = _measure(ex, 2)
        ez_o = _measure(ez, 3)

        vars_map = {
            'vx': vx_o,
            'vz': vz_o,
            'ex': ex_o,
            'ez': ez_o,
            'vx_full': U,
            'vz_full': V,
            'ex_full': ex,
            'ez_full': ez,
            'es_full': es
        }

        if return_vars is not None:
            return new_carry, tuple(vars_map[k] for k in return_vars)

        rec_data = (vx_o, vz_o, ex_o, ez_o)
        if return_wavefields:
            return new_carry, rec_data + (U, V, ex, ez)
        else:
            return new_carry, rec_data

    # Block Checkpointing logic
    num_blocks = nt // block_size
    remainder = nt % block_size

    @jax.checkpoint
    def run_block(carry, block_its):
        return lax.scan(time_step, carry, block_its)

    # Process blocks
    if num_blocks > 0:
        main_its = jnp.arange(num_blocks * block_size).reshape(num_blocks, block_size)
        carry, main_results = lax.scan(run_block, initial_carry, main_its)

        main_results = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), main_results)
    else:
        carry = initial_carry
        _, sample_results = time_step(initial_carry, 0)
        main_results = jax.tree_util.tree_map(lambda x: jnp.zeros((0,) + x.shape), sample_results)

    # Process remainder
    if remainder > 0:
        remainder_its = jnp.arange(num_blocks * block_size, nt)
        carry, remainder_results = lax.scan(time_step, carry, remainder_its)
        results = jax.tree_util.tree_map(lambda r1, r2: jnp.concatenate([r1, r2], axis=0), main_results, remainder_results)
    else:
        results = main_results

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Build JIT-compiled forward function
# ─────────────────────────────────────────────────────────────────────────────

def build_forward_fn(nz, nx, dx, dz, dt, nt, freq, pad, block_size, rec_x, rec_z,
                     fd_order=2, return_vars=None, free_surface=True,
                     gauge_length=None, component=None):
    """
    Build a JIT-compiled single-shot forward function.

    Fixes grid/wavelet params into forward_jax, leaving only
    (Vs, Vp, rho, src_x, src_z, src_wavelet) as free args.

    Parameters
    ----------
    return_vars : list of str, optional
        e.g. ['ex_full'] to return full wavefield for illumination.
    free_surface : bool
        If True, free surface at top (z=0). If False, PML on all 4 sides.
    gauge_length : float, optional
        DAS gauge length L in metres. If set, the inverted `component` is
        moving-averaged over L along x at GRID resolution (dx) and output at every
        physical-domain node (nx_dom DAS channels); the other components are still
        point-sampled at the receivers. If None (default), point strain is
        recorded for all components (no gauge effect).
    component : int, optional
        Which component the gauge applies to (0=vx, 1=vz, 2=exx, 3=ezz). Only this
        component is moving-averaged when gauge_length is set; required for DAS
        mode to know which field is the cable measurement. Ignored if gauge off.

    Returns:
        run_shot(Vs, Vp, rho, src_x, src_z, src_wavelet) → tuple
    """
    _fwd = partial(
        forward_jax,
        nx_dom=nx, nz_dom=nz,
        dx=dx,     dz=dz,
        dt=dt,     nt=nt,
        fc=freq,
        pad=pad,   block_size=block_size,
        fd_order=fd_order,
        return_vars=return_vars,
        free_surface=free_surface,
        gauge_length=gauge_length,
        component=component,
    )
    rec_x_jax = jnp.array(rec_x, dtype=jnp.int32)
    rec_z_jax = jnp.array(rec_z, dtype=jnp.int32)

    def run_shot(Vs, Vp, rho, src_x, src_z, src_wavelet):
        return _fwd(Vs, Vp, rho, src_x, src_z, rec_x_jax, rec_z_jax,
                    src_wavelet=src_wavelet)

    return jax.jit(run_shot)
