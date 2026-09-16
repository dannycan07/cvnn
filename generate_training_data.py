"""
Generate synthetic quantum dynamics training data for all systems in the CVNN project.

Solves the Lindblad master equation:
    dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

Output format per .npy file:
  - Spin-Boson: shape (401, 5),  columns [t, ρ₁₁, ρ₁₂, ρ₂₁, ρ₂₂]
  - N-site FMO: shape (T, 1+N*(N+1)), upper-triangle elements stored at the
    column indices the data-prep scripts use (see pack_trajectory()).

Usage:
    python generate_training_data.py              # all systems, default counts
    python generate_training_data.py --sb 500 --fmo4 200 --fmo7 100 --fmo8 100
"""

import argparse
import os
import numpy as np

HBAR_CM_FS = 5309.1   # ħ in cm⁻¹·fs (energy × time units for FMO)

# ---------------------------------------------------------------------------
# Core ODE solver
# ---------------------------------------------------------------------------

def lindblad_rhs(rho, H, L_ops):
    drho = -1j * (H @ rho - rho @ H)
    for L in L_ops:
        Ld = L.conj().T
        drho += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)
    return drho


def rk4_step(rho, H, L_ops, dt):
    k1 = lindblad_rhs(rho,            H, L_ops)
    k2 = lindblad_rhs(rho + 0.5*dt*k1, H, L_ops)
    k3 = lindblad_rhs(rho + 0.5*dt*k2, H, L_ops)
    k4 = lindblad_rhs(rho + dt*k3,     H, L_ops)
    return rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_lindblad(H, L_ops, rho0, n_steps, dt):
    """Propagate density matrix for n_steps steps of size dt. Returns (n_steps+1, n, n) array."""
    n = H.shape[0]
    traj = np.zeros((n_steps + 1, n, n), dtype=complex)
    traj[0] = rho0.copy()
    rho = rho0.copy().astype(complex)
    for i in range(n_steps):
        rho = rk4_step(rho, H, L_ops, dt)
        # Enforce Hermiticity (suppress numerical drift)
        rho = 0.5 * (rho + rho.conj().T)
        traj[i + 1] = rho
    return traj


# ---------------------------------------------------------------------------
# Raw file packing
# ---------------------------------------------------------------------------

def pack_trajectory(rho_traj, t_array, n_states):
    """
    Pack a (T, n, n) density-matrix trajectory into the raw .npy column format.

    Column layout (0-indexed):
      col 0  : time
      row i of upper triangle starts at col  1 + i*(n_states+1)
              and holds  ρ[i,i], ρ[i,i+1], ..., ρ[i,n_states-1]
              (n_states-i elements)

    Total columns: 1 + n_states*(n_states+1)
    """
    T = rho_traj.shape[0]
    n = n_states
    n_cols = 1 + n * (n + 1)
    raw = np.zeros((T, n_cols), dtype=complex)
    raw[:, 0] = t_array

    for i in range(n):
        col_start = 1 + i * (n + 1)
        for j_off, j in enumerate(range(i, n)):
            raw[:, col_start + j_off] = rho_traj[:, i, j]

    return raw


# ---------------------------------------------------------------------------
# Spin-Boson
# ---------------------------------------------------------------------------

def generate_spinboson_trajectory(rng, dt=0.2, n_steps=400):
    """
    H = (ε/2)σ_z + (Δ/2)σ_x  (ħ = 1)
    Lindblad: relaxation L₁=√γ σ₋, dephasing L₂=√(γ_φ/2) σ_z
    """
    eps   = rng.uniform(-1.0, 1.0)
    Delta = rng.uniform(0.3, 1.5)
    gamma = rng.uniform(0.01, 0.25)
    gamma_phi = rng.uniform(0.01, 0.20)

    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sm = np.array([[0, 0], [1, 0]], dtype=complex)   # σ₋

    H = 0.5 * eps * sz + 0.5 * Delta * sx
    L_ops = [np.sqrt(gamma) * sm,
             np.sqrt(gamma_phi / 2.0) * sz]

    # Random initial pure state (varied excitation)
    theta = rng.uniform(0, np.pi)
    phi   = rng.uniform(0, 2 * np.pi)
    psi   = np.array([np.cos(theta / 2),
                      np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)
    rho0  = np.outer(psi, psi.conj())

    rho_traj = integrate_lindblad(H, L_ops, rho0, n_steps, dt)
    t_array  = np.arange(n_steps + 1) * dt

    # Pack into (401, 5): [t, ρ₁₁, ρ₁₂, ρ₂₁, ρ₂₂]
    raw = np.zeros((n_steps + 1, 5), dtype=complex)
    raw[:, 0] = t_array
    raw[:, 1] = rho_traj[:, 0, 0]   # ρ₁₁
    raw[:, 2] = rho_traj[:, 0, 1]   # ρ₁₂
    raw[:, 3] = rho_traj[:, 1, 0]   # ρ₂₁
    raw[:, 4] = rho_traj[:, 1, 1]   # ρ₂₂
    return raw


# ---------------------------------------------------------------------------
# FMO helpers
# ---------------------------------------------------------------------------

# Adolphs & Renger (2006) 7-site FMO Hamiltonian (cm⁻¹)
H_FMO7_BASE = np.array([
    [12410, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],
    [-87.7, 12530,  30.8,   8.2,   0.7,  11.8,   4.3],
    [  5.5,  30.8, 12210, -53.5,  -2.2,  -9.6,   6.0],
    [ -5.9,   8.2, -53.5, 12320, -70.7, -17.0, -63.3],
    [  6.7,   0.7,  -2.2, -70.7, 12480,  81.1,  -1.3],
    [-13.7,  11.8,  -9.6, -17.0,  81.1, 12630,  39.7],
    [ -9.9,   4.3,   6.0, -63.3,  -1.3,  39.7, 12440],
], dtype=float)

# 4-site hypothetical FMO (sites 1–4 from 7-site + realistic coupling)
H_FMO4_BASE = np.array([
    [12410, -87.7,   5.5,  -5.9],
    [-87.7, 12530,  30.8,   8.2],
    [  5.5,  30.8, 12210, -53.5],
    [ -5.9,   8.2, -53.5, 12320],
], dtype=float)

# 8-site FMO (extend 7-site with an 8th chromophore, representative values)
_E8 = 12380.0
_C8 = np.array([-12.5, 3.1, -8.2, 15.4, -6.0, 9.7, 22.3])
H_FMO8_BASE = np.zeros((8, 8), dtype=float)
H_FMO8_BASE[:7, :7] = H_FMO7_BASE
H_FMO8_BASE[7, 7] = _E8
H_FMO8_BASE[:7, 7] = _C8
H_FMO8_BASE[7, :7] = _C8


def perturb_hamiltonian(H_base, rng, diag_noise=30.0, off_noise=15.0):
    """Add Gaussian noise to a Hamiltonian (cm⁻¹)."""
    n = H_base.shape[0]
    noise = np.zeros_like(H_base)
    for i in range(n):
        noise[i, i] = rng.normal(0, diag_noise)
        for j in range(i + 1, n):
            v = rng.normal(0, off_noise)
            noise[i, j] = v
            noise[j, i] = v
    return H_base + noise


def fmo_lindblad_ops(n, dephasing_rates_cm):
    """Site-local dephasing: L_k = √(γ_k / ħ) |k⟩⟨k|  (rates in cm⁻¹ / ħ → rad/fs)."""
    ops = []
    for k, gamma_cm in enumerate(dephasing_rates_cm):
        gamma_rad_fs = gamma_cm / HBAR_CM_FS
        L = np.zeros((n, n), dtype=complex)
        L[k, k] = np.sqrt(gamma_rad_fs)
        ops.append(L)
    return ops


def generate_fmo_trajectory(H_base, rng, n_steps=400, dt_fs=2.0,
                             init_site=None, dephasing_lo=100.0, dephasing_hi=300.0):
    """
    Generate one FMO trajectory.
      H_base   : (n, n) Hamiltonian in cm⁻¹
      dt_fs    : timestep in femtoseconds
      init_site: which site to start in (None → random)
    """
    n = H_base.shape[0]
    H_cm = perturb_hamiltonian(H_base, rng)
    H    = H_cm / HBAR_CM_FS                       # convert to rad/fs (ħ=1)

    deph_rates = rng.uniform(dephasing_lo, dephasing_hi, size=n)
    L_ops = fmo_lindblad_ops(n, deph_rates)

    if init_site is None:
        init_site = rng.integers(0, n)
    rho0 = np.zeros((n, n), dtype=complex)
    rho0[init_site, init_site] = 1.0

    rho_traj = integrate_lindblad(H, L_ops, rho0, n_steps, dt_fs)
    t_array  = np.arange(n_steps + 1) * dt_fs

    return pack_trajectory(rho_traj, t_array, n)


# ---------------------------------------------------------------------------
# Top-level generators
# ---------------------------------------------------------------------------

def generate_system(system, n_traj, out_dir, rng):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Generating {n_traj} trajectories -> {out_dir}")

    for i in range(n_traj):
        if system == "sb":
            raw = generate_spinboson_trajectory(rng)
        elif system == "fmo4":
            raw = generate_fmo_trajectory(H_FMO4_BASE, rng, n_steps=400, dt_fs=2.0)
        elif system == "fmo7":
            raw = generate_fmo_trajectory(H_FMO7_BASE, rng, n_steps=480, dt_fs=2.0)
        elif system == "fmo8":
            raw = generate_fmo_trajectory(H_FMO8_BASE, rng, n_steps=400, dt_fs=2.0)

        path = os.path.join(out_dir, f"traj_{i:04d}.npy")
        np.save(path, raw)

        if (i + 1) % max(1, n_traj // 10) == 0:
            print(f"    {i + 1}/{n_traj} done")

    print(f"  Done. Shape of last file: {raw.shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate CVNN training trajectories")
    parser.add_argument("--sb",   type=int, default=500, help="# Spin-Boson trajectories")
    parser.add_argument("--fmo4", type=int, default=200, help="# 4-site FMO trajectories")
    parser.add_argument("--fmo7", type=int, default=100, help="# 7-site FMO trajectories")
    parser.add_argument("--fmo8", type=int, default=100, help="# 8-site FMO trajectories")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed")
    parser.add_argument("--out",  type=str, default="training_data",
                        help="Root output directory (default: training_data/)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.out

    configs = [
        ("sb",   args.sb,   os.path.join(root, "spinboson", "combined")),
        ("fmo4", args.fmo4, os.path.join(root, "fmo4")),
        ("fmo7", args.fmo7, os.path.join(root, "fmo7", "init_1")),
        ("fmo8", args.fmo8, os.path.join(root, "fmo8")),
    ]

    for system, n_traj, out_dir in configs:
        if n_traj > 0:
            print(f"\n[{system.upper()}]")
            generate_system(system, n_traj, out_dir, rng)

    print("\nAll done. Update data_dir in each prep_*.py to point at these directories.")
    print(f"  Spin-Boson : {os.path.join(root, 'spinboson', 'combined', '*.npy')}")
    print(f"  4-site FMO : {os.path.join(root, 'fmo4', '*.npy')}")
    print(f"  7-site FMO : {os.path.join(root, 'fmo7', 'init_1', '*.npy')}")
    print(f"  8-site FMO : {os.path.join(root, 'fmo8', '*.npy')}")


if __name__ == "__main__":
    main()
