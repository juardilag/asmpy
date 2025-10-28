import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy.fft as jfft
import matplotlib.pyplot as plt

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