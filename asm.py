import jax
import jax.numpy as jnp
import numpy as np
import jax.numpy.fft as jfft
from functools import partial

# Pulse Initialation

def initialize_gaussian_pulse_3d(N_t, N_x, L, T, w0, tau, chirp=0.0):
    """
    Initializes a 3D spatiotemporal Gaussian pulse.

    Args:
        N_t, N_x: Grid points in time and space (N_y assumed equal to N_x).
        L: Spatial box size.
        T: Temporal window size.
        w0: Beam waist radius (spatial width).
        tau: Pulse duration (temporal width).
        chirp: Linear chirp parameter (C).

    Returns:
        jnp.ndarray: The 3D complex field envelope E(t, x, y).
    """
    # Grid setup
    t = jnp.linspace(-T/2, T/2, N_t)
    x = jnp.linspace(-L/2, L/2, N_x)
    X, Y = jnp.meshgrid(x, x, indexing='ij')

    # Temporal Gaussian with Chirp: exp(-(1+iC) * t^2 / (2*tau^2))
    # We use broadcasting: t shape (N_t, 1, 1)
    temporal_profile = jnp.exp(-(1 + 1j * chirp) * (t[:, None, None]**2) / (2 * tau**2))

    # Spatial Gaussian: exp(-(x^2 + y^2) / w0^2)
    # Shape (1, N_x, N_x)
    r_squared = X**2 + Y**2
    spatial_profile = jnp.exp(-r_squared / (w0**2))
    spatial_profile = spatial_profile[None, :, :]

    return temporal_profile * spatial_profile

def normalize_field_to_energy(E_txy, energy_joules, dx, dt, n=1.0):
    """
    Rescales a field array E_txy so that its total energy equals 'energy_joules'.
    
    Args:
        E_txy: The input complex field (arbitrary amplitude).
        energy_joules: Desired pulse energy (e.g., 1e-9 for 1 nJ).
        dx: Spatial grid step (meters).
        dt: Time grid step (seconds).
        n: Refractive index.
        
    Returns:
        jax.Array: The rescaled field in units of [V/m].
    """
    c = 299792458.0
    epsilon0 = 8.8541878128e-12
    
    # 1. Calculate current "unscaled" energy in the grid
    # Energy = Sum( Intensity * dA * dt )
    # Intensity = 0.5 * c * n * eps0 * |E|^2
    
    intensity_unscaled = 0.5 * c * n * epsilon0 * jnp.abs(E_txy)**2
    total_energy_current = jnp.sum(intensity_unscaled) * (dx**2) * dt
    
    # 2. Calculate Scaling Factor
    # We need: total_energy_current * scale^2 = energy_joules
    scale_factor = jnp.sqrt(energy_joules / total_energy_current)
    
    print(f"Rescaling input field by factor: {scale_factor:.2e}")
    return E_txy * scale_factor

# ASM implementation

@jax.jit
def propagate_spatial_asm(E_in, z, L, wavelength, n=1.0):
    """
    Propagates a 2D electric field using the angular spectrum method.

    This function correctly handles the FFT of a centered field by using
    ifftshift before the FFT and fftshift after the IFFT.

    Args:
        E_in (jax.Array): The input 2D electric field at z=0 (N x N array).
        z (float): The propagation distance (in meters).
        L (float): The physical side length of the simulation grid (in meters).
        wavelength (float): The vacuum wavelength of the light (in meters).
        n (float): The refractive index of the medium (default 1.0).

    Returns:
        jax.Array: The propagated 2D electric field at distance z.
    """
    N = E_in.shape[0]
    if E_in.shape[0] != E_in.shape[1]:
        raise ValueError("Input field E_in must be a square (N x N) array.")

    # 1. Get Spatial and Frequency Grid Parameters
    dx = L / N  # Spatial grid spacing
    k = 2 * jnp.pi * n / wavelength  # Wave number in the medium

    # 2. Create k-space grid
    # Get frequency coordinates (as produced by fftfreq)
    # This order [0, 1, ..., N/2-1, -N/2, ..., -1] * (2*pi/L)
    # correctly matches the output layout of jfft.fft2
    k_vec = 2 * jnp.pi * jfft.fftfreq(N, d=dx)
    
    # Create 2D k-space grid
    # indexing='ij' ensures Kx and Ky have shape (N, N)
    Kx, Ky = jnp.meshgrid(k_vec, k_vec, indexing='ij')

    # 3. Calculate k_z
    # k_z^2 = k^2 - k_x^2 - k_y^2
    k_z_squared = k**2 - Kx**2 - Ky**2

    # Handle evanescent waves (where k_z_squared < 0)
    # We cast to complex *before* the sqrt. This correctly results in
    # a purely imaginary k_z (e.g., sqrt(-4+0j) = 2j),
    # which leads to exponential decay (exp(1j * (2j) * z) = exp(-2z)).
    k_z = jnp.sqrt(k_z_squared.astype(jnp.complex128))

    # 4. Define Propagation Kernel (Transfer Function)
    # H(kx, ky) = exp(i * k_z * z)
    H = jnp.exp(1j * k_z * z)

    # 5. Perform the propagation
    
    # E_in is centered in the spatial grid (e.g., at index N/2, N/2)
    # We must use ifftshift to move the center to index (0, 0)
    # which is what the FFT algorithm assumes is the origin.
    E_in_shifted = jfft.ifftshift(E_in)

    # Compute the angular spectrum (Fourier transform)
    A_in = jfft.fft2(E_in_shifted)

    # Apply the propagation kernel in k-space
    A_out = A_in * H

    # Inverse Fourier transform to get back to spatial domain
    E_out_shifted = jfft.ifft2(A_out)

    # Shift the (0, 0) origin back to the center (N/2, N/2) for
    # correct spatial representation.
    E_out = jfft.fftshift(E_out_shifted)

    return E_out


def propagate_asm_spatiotemporal(E_in_txy, z, L, T, wavelength_0, n_func=None):
    """
    Propagates a 3D spatiotemporal *envelope* field (t, x, y)
    using the angular spectrum method.

    This function assumes E_in_txy is the *complex envelope* and
    its spectrum is centered at omega=0 (not omega_0).

    Args:
        E_in_txy (jax.Array): Input 3D field envelope, shape (N_t, N_x, N_y).
        z (float): Propagation distance (in meters).
        L (float): Physical side length of spatial grid (in meters).
        T (float): Total time window (in seconds).
        wavelength_0 (float): The *central* vacuum wavelength (in meters).
        n_func (callable, optional): A function n(omega) that returns the
                                     refractive index for a given *angular
                                     frequency offset* (delta_omega).
                                     If None, assumes n=1.0 (vacuum).

    Returns:
        (jax.Array, jax.Array): (E_out_txy, t)
                                E_out_txy: Propagated 3D field envelope (N_t, N_x, N_y)
                                t: The time vector for plotting
    """
    c = 299792458.0  # Speed of light
    omega_0 = 2 * jnp.pi * c / wavelength_0
    
    # Default to vacuum, n=1.0 for all frequencies
    if n_func is None:
        n_func = lambda delta_omega: 1.0

    # Get n at the central frequency.
    n_0 = n_func(0.0)
    k_0 = n_0 * omega_0 / c # Central wavenumber

    N_t, N_x, N_y = E_in_txy.shape
    if N_x != N_y:
        raise ValueError("Input field E_in must be square in space (N_x == N_y).")
    
    dx = L / N_x # Spatial grid spacing

    # 1. Get time and frequency grids
    dt = T / N_t
    t = jnp.linspace(-T/2, T/2, N_t)
    
    # Get angular frequency *offset* (delta_omega)
    # This vector is centered at 0.
    nu_vec = jfft.fftfreq(N_t, d=dt)
    delta_omega_vec = 2 * jnp.pi * nu_vec  # (rad/s)

    # 2. Perform 1D FFT over time
    # E_in_txy is an envelope centered at t=0
    E_in_t_shifted = jfft.ifftshift(E_in_txy, axes=0)
    E_tilde_delta_omega_xy = jfft.fft(E_in_t_shifted, axis=0)
    
    # E_tilde_delta_omega_xy[i, :, :] now corresponds to delta_omega_vec[i]

    # 3. Create k-space grid (spatial)
    k_vec_spatial = 2 * jnp.pi * jfft.fftfreq(N_x, d=dx)
    Kx, Ky = jnp.meshgrid(k_vec_spatial, k_vec_spatial, indexing='ij')

    # 4. Define the (vmapped) propagation function
    
    @partial(jax.jit, static_argnums=(3,))
    def _propagate_asm_single_freq_vmap(E_in_xy, delta_omega, z, L):
        """
        Internal JIT-compiled function for vmapping.
        Propagates a single (delta_omega) slice.
        """
        # Get frequency-dependent parameters
        omega = omega_0 + delta_omega
        n = n_func(delta_omega) # n is a function of the offset
        k = n * omega / c       # Full k for this frequency
        
        # Calculate k_z, handling evanescent waves
        k_z_squared = k**2 - Kx**2 - Ky**2
        k_z = jnp.sqrt(k_z_squared.astype(jnp.complex128))

        # --- THIS IS THE KEY FIX ---
        # We use the envelope propagator kernel: H_env = exp(i * (k_z - k_0) * z)
        # This removes the large, fast-varying phase (k_0 * z)
        # and only propagates the *difference* relative to the center.
        H_env = jnp.exp(-1j * (k_z - k_0) * z)
        
        # Standard ASM propagation
        E_in_shifted_xy = jfft.ifftshift(E_in_xy)
        A_in = jfft.fft2(E_in_shifted_xy)
        A_out = A_in * H_env # Apply envelope kernel
        E_out_shifted_xy = jfft.ifft2(A_out)
        E_out_xy = jfft.fftshift(E_out_shifted_xy)
        return E_out_xy

    # 5. Run the vectorized propagation
    # vmap over axis 0 of E_tilde_delta_omega_xy
    # vmap over axis 0 of delta_omega_vec
    # 'z' and 'L' are constants (None)
    E_tilde_out_delta_omega_xy = jax.vmap(
        _propagate_asm_single_freq_vmap, 
        in_axes=(0, 0, None, None), 
        out_axes=0
    )(E_tilde_delta_omega_xy, delta_omega_vec, z, L)

    # 6. Perform 1D IFFT over time to get back
    E_out_t_shifted = jfft.ifft(E_tilde_out_delta_omega_xy, axis=0)
    E_out_txy = jfft.fftshift(E_out_t_shifted, axes=0)
    
    return E_out_txy


# Second Harmonic Generation Simulation

def get_asm_phase_masks(N_t, N_x, L, T, wavelength, n_func, dz):
    """
    Pre-calculates the linear propagation phase mask H(kx, ky, delta_omega)
    for a single step of distance dz.
    """
    c = 299792458.0
    omega_0 = 2 * jnp.pi * c / wavelength
    
    # 1. Frequency Grid (Offset from carrier)
    dt = T / N_t
    nu_vec = jfft.fftfreq(N_t, d=dt)
    delta_omega = 2 * jnp.pi * nu_vec
    
    # 2. Spatial Grid
    dx = L / N_x
    k_vec_spatial = 2 * jnp.pi * jfft.fftfreq(N_x, d=dx)
    Kx, Ky = jnp.meshgrid(k_vec_spatial, k_vec_spatial, indexing='ij')

    # 3. Calculate H for every frequency slice
    # This uses the same logic as your propagate_asm_spatiotemporal
    
    def _get_slice_kernel(d_w):
        omega = omega_0 + d_w
        n = n_func(d_w)
        k = n * omega / c
        k_0 = n_func(0.0) * omega_0 / c
        
        k_z_sq = k**2 - Kx**2 - Ky**2
        k_z = jnp.sqrt(k_z_sq.astype(jnp.complex128))
        
        # Envelope propagator: removes the carrier phase k_0 * z
        return jnp.exp(1j * (k_z - k_0) * dz)

    # Vectorize over frequencies
    H_cube = jax.vmap(_get_slice_kernel)(delta_omega)
    
    # Shape: (N_t, N_x, N_y)
    return H_cube

# --- THE LINEAR STEP ---
@jax.jit
def apply_linear_step(E_txy, H_cube):
    """
    Applies the pre-calculated ASM phase mask.
    E_txy: (N_t, N_x, N_y)
    H_cube: (N_t, N_x, N_y) Pre-computed exp(i*kz*dz)
    """
    # FFT Time -> Freq
    E_w = jfft.fft(E_txy, axis=0)
    
    # Shift Spatial -> k-space (using fft2)
    # Note: We need ifftshift on input for standard FFT center handling
    # but since we act on the whole cube, we can be careful.
    # Let's stick to the robust shift-fft-shift method from your code.
    
    # 1. Shift time center to origin
    E_w_shifted = jfft.ifftshift(E_w, axes=0) 
    
    # 2. Shift space center to origin
    E_w_space_shifted = jfft.ifftshift(E_w_shifted, axes=(1,2))
    
    # 3. Spatial FFT
    A_in = jfft.fft2(E_w_space_shifted, axes=(1,2))
    
    # 4. Apply Mask
    A_out = A_in * H_cube
    
    # 5. Inverse Transforms
    E_out_space_shifted = jfft.ifft2(A_out, axes=(1,2))
    E_out_shifted = jfft.fftshift(E_out_space_shifted, axes=(1,2))
    E_txy_out = jfft.ifft(jfft.fftshift(E_out_shifted, axes=0), axis=0)
    
    return E_txy_out

# --- THE NONLINEAR STEP (RK4) ---
@jax.jit
def apply_nonlinear_step_rk4(E1, E2, dz, omega1, omega2, n1, n2, d_eff):
    """
    Solves the coupled ODEs for SHG over distance dz using RK4.
    Field 1: Fundamental (omega)
    Field 2: Second Harmonic (2*omega)
    """
    c = 299792458.0
    
    # Coupling coefficients
    # kappa1 units: m/V * rad/s / (m/s) = 1/V. Field is V/m. Result is 1/m. Correct.
    kappa1 = (omega1 * d_eff) / (n1 * c)
    kappa2 = (omega2 * d_eff) / (n2 * c)

    def ode_system(fields):
        A1, A2 = fields
        # Coupled Wave Equations
        # dA1/dz = i * kappa1 * A2 * conj(A1)
        # dA2/dz = i * kappa2 * A1^2
        dA1_dz = 1j * kappa1 * A2 * jnp.conj(A1)
        dA2_dz = 1j * kappa2 * (A1**2)
        return (dA1_dz, dA2_dz)

    # RK4 Integration
    # k1
    k1_A1, k1_A2 = ode_system((E1, E2))
    
    # k2
    E1_k2 = E1 + 0.5 * dz * k1_A1
    E2_k2 = E2 + 0.5 * dz * k1_A2
    k2_A1, k2_A2 = ode_system((E1_k2, E2_k2))
    
    # k3
    E1_k3 = E1 + 0.5 * dz * k2_A1
    E2_k3 = E2 + 0.5 * dz * k2_A2
    k3_A1, k3_A2 = ode_system((E1_k3, E2_k3))
    
    # k4
    E1_k4 = E1 + dz * k3_A1
    E2_k4 = E2 + dz * k3_A2
    k4_A1, k4_A2 = ode_system((E1_k4, E2_k4))
    
    # Final Update
    E1_new = E1 + (dz / 6.0) * (k1_A1 + 2*k2_A1 + 2*k3_A1 + k4_A1)
    E2_new = E2 + (dz / 6.0) * (k1_A2 + 2*k2_A2 + 2*k3_A2 + k4_A2)
    
    return E1_new, E2_new

# --- MAIN SSFM LOOP ---
def run_shg_simulation_with_history(
    E1_in, E2_in, 
    L, T, 
    lambda1, 
    n1_func, n2_func, 
    d_eff, 
    z_total, N_steps
):
    """
    Runs SHG simulation and returns field evolution history.
    
    Returns:
        E1_final, E2_final: The 3D fields at the end of the crystal.
        U1_hist, U2_hist:   1D arrays containing the Total Energy vs z.
    """
    # 1. Setup Constants
    dz = z_total / N_steps
    c = 299792458.0
    omega1 = 2 * jnp.pi * c / lambda1
    omega2 = 2 * omega1
    lambda2 = lambda1 / 2.0
    
    N_t, N_x, _ = E1_in.shape
    
    # 2. Pre-calculate Linear Propagators (Phase Masks)
    # We use Strang Splitting: Linear(dz/2) -> Nonlin(dz) -> Linear(dz/2)
    H1_half = get_asm_phase_masks(N_t, N_x, L, T, lambda1, n1_func, dz/2)
    H2_half = get_asm_phase_masks(N_t, N_x, L, T, lambda2, n2_func, dz/2)
    
    # Get refractive indices at center frequency for the nonlinear coupling
    n1_center = n1_func(0.0)
    n2_center = n2_func(0.0)

    # 3. Define the Stepping Function
    def step_fn(carry, _):
        E1, E2 = carry
        
        # --- A. Linear Half-Step ---
        E1 = apply_linear_step(E1, H1_half)
        E2 = apply_linear_step(E2, H2_half)
        
        # --- B. Nonlinear Full-Step (RK4) ---
        E1, E2 = apply_nonlinear_step_rk4(
            E1, E2, dz, omega1, omega2, n1_center, n2_center, d_eff
        )
        
        # --- C. Linear Half-Step ---
        E1 = apply_linear_step(E1, H1_half)
        E2 = apply_linear_step(E2, H2_half)
        
        # --- D. Calculate Energy (History Tracking) ---
        # We sum |E|^2. This is proportional to Energy. 
        # (To get Joules, one would multiply by dx*dy*dt*constants later)
        U1 = jnp.sum(jnp.abs(E1)**2)
        U2 = jnp.sum(jnp.abs(E2)**2)
        
        # Return new state AND data to stack
        return (E1, E2), (U1, U2)

    # 4. Run the Loop (Scan)
    (E1_out, E2_out), (U1_hist, U2_hist) = jax.lax.scan(
        step_fn,          # Function to apply
        (E1_in, E2_in),   # Initial state
        None,             # Inputs to loop (None because our operators are constant)
        length=N_steps    # Number of iterations
    )
    
    return E1_out, E2_out, U1_hist, U2_hist