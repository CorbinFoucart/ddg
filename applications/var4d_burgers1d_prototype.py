r"""1D 4D-VAR twin experiment on viscous Burgers -- methodology validation.

Continuous model (periodic on [0, L)):
    u_t + u u_x = nu u_xx .

Discrete forward map  M : u_0 -> {u(t_k)}  is an explicit pseudo-spectral
RK4 march (FFT derivatives, 2/3 dealiasing).  Observations are point
samples of u at  n_probe  locations and  n_obs  times, generated from a
known truth  u_0^true  (a "twin experiment").  4D-VAR minimises

    J(u_0) = 1/2 sum_k || H_k M(u_0) - y_k ||^2  +  lambda/2 || u_0 - u_b ||^2,

with background u_b = 0.  The gradient dJ/du_0 is the discrete adjoint,
obtained here verbatim by reverse-mode AD through the whole RK4 trajectory
(jax.grad) -- no hand-derived adjoint.  Adam descends J.

This is the validation rig: with a rich-enough (probe x time) budget the
inverse problem is well posed and u_0 is recovered to a small relative
error; starving the budget reproduces the underdetermined regime.

  python scripts/var4d_burgers1d_prototype.py --grad-check
  python scripts/var4d_burgers1d_prototype.py --n-probe 32 --n-obs 20
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def make_model(N, L, nu):
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)          # wavenumbers
    k = jnp.asarray(k)
    ik = 1j * k
    k2 = k * k
    # 2/3 dealiasing mask
    kmax = (2.0 / 3.0) * (np.pi * N / L)
    mask = jnp.asarray(np.abs(np.asarray(k)) <= kmax, dtype=jnp.float64)

    def rhs(u):
        uh = jnp.fft.fft(u)
        ux = jnp.real(jnp.fft.ifft(ik * uh))
        uxx = jnp.real(jnp.fft.ifft(-k2 * uh))
        nl = jnp.real(jnp.fft.ifft(mask * jnp.fft.fft(u * ux)))   # dealiased
        return -nl + nu * uxx

    def rk4(u, dt):
        k1 = rhs(u)
        k2_ = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2_)
        k4 = rhs(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2 * k2_ + 2 * k3 + k4)

    return rk4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--L", type=float, default=2.0 * np.pi)
    ap.add_argument("--nu", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=2.0e-3)
    ap.add_argument("--tfinal", type=float, default=1.0)
    ap.add_argument("--n-probe", type=int, default=32, help="spatial probes")
    ap.add_argument("--n-obs", type=int, default=20, help="observation times")
    ap.add_argument("--noise", type=float, default=0.0, help="obs noise std")
    ap.add_argument("--lam-bg", type=float, default=1.0e-4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-check", action="store_true")
    args = ap.parse_args()

    rk4 = make_model(args.N, args.L, args.nu)
    nt = int(round(args.tfinal / args.dt))
    x = np.linspace(0.0, args.L, args.N, endpoint=False)

    # --- truth IC: a few smooth modes (fixed) ---
    rng = np.random.default_rng(args.seed)
    u0_true = (np.sin(x) + 0.5 * np.sin(2 * x + 0.7)
               + 0.3 * np.cos(3 * x - 0.4))
    u0_true = jnp.asarray(u0_true)

    obs_steps = np.unique(np.linspace(1, nt, args.n_obs).astype(int))
    probe_idx = jnp.asarray(np.linspace(0, args.N, args.n_probe,
                                        endpoint=False).astype(int))
    obs_set = set(int(s) for s in obs_steps)
    obs_steps_j = jnp.asarray(sorted(obs_set))

    @jax.jit
    def trajectory_obs(u0):
        def body(carry, step):
            u = rk4(carry, args.dt)
            return u, u[probe_idx]
        # scan over all steps, collect probe values at every step, then pick
        steps = jnp.arange(1, nt + 1)
        _, allobs = jax.lax.scan(body, u0, steps)        # (nt, n_probe)
        return allobs[obs_steps_j - 1]                   # (n_obs, n_probe)

    y_clean = trajectory_obs(u0_true)
    y_obs = y_clean
    if args.noise > 0:
        y_obs = y_clean + args.noise * jnp.asarray(
            rng.standard_normal(np.asarray(y_clean).shape))

    u_b = jnp.zeros(args.N)

    @jax.jit
    def loss(u0):
        pred = trajectory_obs(u0)
        data = 0.5 * jnp.sum((pred - y_obs) ** 2)
        bg = 0.5 * args.lam_bg * jnp.sum((u0 - u_b) ** 2)
        return data + bg

    grad_fn = jax.jit(jax.grad(loss))

    def relerr(u0):
        return float(jnp.linalg.norm(u0 - u0_true) / jnp.linalg.norm(u0_true))

    print(f"[var4d-burgers] N={args.N} nu={args.nu} dt={args.dt} nt={nt} "
          f"probes={int(probe_idx.size)} obs_times={int(obs_steps_j.size)} "
          f"(obs dofs={int(probe_idx.size*obs_steps_j.size)} vs N={args.N})",
          flush=True)

    if args.grad_check:
        u_test = u_b + 0.1 * jnp.asarray(rng.standard_normal(args.N))
        g = grad_fn(u_test)
        h = 1e-4
        for i in (0, args.N // 3, 2 * args.N // 3):
            ep = u_test.at[i].add(h); em = u_test.at[i].add(-h)
            gfd = float((loss(ep) - loss(em)) / (2 * h)); gad = float(g[i])
            print(f"  d/du0[{i:3d}]: AD={gad:+.6e} FD={gfd:+.6e} "
                  f"rel={abs(gad-gfd)/(abs(gfd)+1e-30):.2e}", flush=True)
        return

    # --- Adam ---
    u0 = u_b
    mt = jnp.zeros_like(u0); vt = jnp.zeros_like(u0)
    b1, b2, eps = 0.9, 0.999, 1e-8
    L0 = float(loss(u0)); hist = []
    for it in range(1, args.n_train + 1):
        Lv, g = float(loss(u0)), grad_fn(u0)
        mt = b1 * mt + (1 - b1) * g; vt = b2 * vt + (1 - b2) * g * g
        u0 = u0 - args.lr * (mt / (1 - b1 ** it)) / (jnp.sqrt(vt / (1 - b2 ** it)) + eps)
        re = relerr(u0)
        hist.append((it, Lv, re))
        if it % 25 == 0 or it == 1:
            print(f"  it {it:4d}: J={Lv:.4e}  rel_IC_err={re:.4e}", flush=True)
    re_final = relerr(u0)
    print(f"\n  final: J {L0:.3e} -> {float(loss(u0)):.3e}   "
          f"rel IC error {re_final:.4e}", flush=True)

    # --- figure: true vs recovered IC + convergence ---
    its = np.array([h[0] for h in hist])
    Ls = np.array([h[1] for h in hist]); res = np.array([h[2] for h in hist])
    plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 13.5,
                         "axes.labelsize": 12.5, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 10.5})
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.2))
    # left: IC recovery (recovered = C0 blue, dashed)
    ax[0].plot(x, np.asarray(u0_true), "-", color="k", lw=1.8, label="true $u_0$")
    ax[0].plot(x, np.asarray(u0), "--", color="C0", lw=1.6,
               label="recovered $u_0$")
    ax[0].plot(x[np.asarray(probe_idx)],
               np.asarray(u0_true)[np.asarray(probe_idx)], "o", mfc="none",
               mec="0.5", ms=3.5, label="probe columns")
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u_0$")
    ax[0].set_title(f"IC recovery (rel err {re_final:.1e})")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    ax[0].set_box_aspect(1.0)
    # right: convergence -- distinguish by MARKER, not colour
    me = max(1, len(its) // 12)
    ax[1].semilogy(its, Ls, "-o", color="k", mfc="white", mec="k", ms=3.2,
                   markevery=me, lw=1.0, label=r"objective $\mathcal{J}$")
    ax[1].semilogy(its, res, "-s", color="k", mfc="white", mec="k", ms=3.2,
                   markevery=(me // 2, me), lw=1.0, label="rel IC error")
    ax[1].set_xlabel("Adam iteration"); ax[1].set_title("convergence")
    ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_box_aspect(1.0)
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout()
    out = ROOT / "output" / "var4d_burgers1d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
