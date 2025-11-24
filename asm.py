import jax
import jax.numpy as jnp
import numpy as np
import jax.numpy.fft as jfft
from functools import partial

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