"""
多维基准对比（可中断续跑）
=========================

对比算法：
1) TuRBO (hetero scoring)
2) 传统 BO: GP + EI

测试函数：
- Ackley
- F4（Rosenbrock 变体，本文实现采用稳定的 '+' 形式）

维度：
- 10, 20, 50

中断续跑：
- 每个任务（函数+维度+算法）都有独立 checkpoint
- Ctrl+C 后可用 --resume 继续
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube
from turbo1 import Turbo1


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    return float(
        -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
        - np.exp(np.sum(np.cos(c * x)) / d)
        + a
        + np.e
    )


def f4_rosen_log(x: np.ndarray) -> float:
    """
    F4: 基于 Rosenbrock 的对数形式。
    采用稳定形式:
        F4(x) = 20 * log( sum_i [100*(x_{i+1}-x_i^2)^2 + (x_i-1)^2] + 1 )
    全局最优在 x=[1,...,1], F4=0.
    """
    x = np.asarray(x, dtype=float)
    t = 100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2
    return float(20.0 * np.log(np.sum(t) + 1.0))


def expected_improvement(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu - xi
    z = imp / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return imp * cdf + sigma * pdf


@dataclass
class RunState:
    X: np.ndarray
    y: np.ndarray
    best_hist: List[float]
    extra: Dict


def save_state(path: str, state: RunState):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def load_state(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_turbo(
    f: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    iters: int,
    seed: int,
    checkpoint: str,
    resume: bool,
):
    dim = len(lb)
    state = load_state(checkpoint) if resume else None
    turbo = Turbo1(
        f=None,
        lb=lb,
        ub=ub,
        n_init=max(2 * dim, 10),
        max_evals=100000,
        batch_size=1,
        verbose=False,
        use_hetero_lcb=1,
        hetero_beta0=2.2,
        hetero_beta1=0.8,
        hetero_noise_penalty=0.35,
        hetero_k_neighbors=min(8, max(4, dim // 4)),
        use_lcb=0,
        budget=iters,
    )

    if state is None:
        X = np.zeros((0, dim))
        y = np.zeros((0, 1))
        best_hist = []
        hypers = {}
    else:
        X, y, best_hist = state.X, state.y, state.best_hist
        hypers = state.extra.get("hypers", {})
        turbo._X = to_unit_cube(X, lb, ub)
        turbo._fX = np.array(y, copy=True)
        turbo.X = np.array(X, copy=True)
        turbo.fX = np.array(y, copy=True)
        turbo.used_budget = len(y)
        print(f"[TuRBO] Resume: {os.path.basename(checkpoint)} from iter={len(y)}")

    try:
        while len(y) < iters:
            if len(y) < turbo.n_init:
                x_unit = latin_hypercube(1, dim).reshape(-1)
            else:
                X_unit = to_unit_cube(X, lb, ub)
                X_cand, y_cand, hypers = turbo._create_candidates(
                    X=X_unit,
                    fX=y.ravel(),
                    length=turbo.length,
                    n_training_steps=50 if dim >= 20 else 70,
                    hypers=hypers,
                    used_budget=len(y),
                )
                x_unit = turbo._select_candidates(X_cand, y_cand)[0]

            x = from_unit_cube(x_unit[None, :], lb, ub).reshape(-1)
            fx = f(x)
            if len(y) >= turbo.n_init:
                turbo._adjust_length(np.array([[fx]]))

            X = np.vstack([X, x[None, :]])
            y = np.vstack([y, [[fx]]])
            turbo._X = np.vstack([turbo._X, x_unit[None, :]]) if len(turbo._X) else x_unit[None, :]
            turbo._fX = np.vstack([turbo._fX, [[fx]]]) if len(turbo._fX) else np.array([[fx]])
            turbo.X = np.vstack([turbo.X, x[None, :]])
            turbo.fX = np.vstack([turbo.fX, [[fx]]])

            best_hist.append(float(np.min(y)))
            save_state(checkpoint, RunState(X=X, y=y, best_hist=best_hist, extra={"hypers": hypers}))
            print(f"[TuRBO] {os.path.basename(checkpoint)} iter={len(y):03d} best={best_hist[-1]:.6f}")
    except KeyboardInterrupt:
        print(f"\n[TuRBO] Interrupted: {checkpoint}")

    return X, y, best_hist


def run_gp_ei(
    f: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    iters: int,
    seed: int,
    checkpoint: str,
    resume: bool,
):
    dim = len(lb)
    rng = np.random.default_rng(seed)
    state = load_state(checkpoint) if resume else None

    if state is None:
        n_init = max(2 * dim, 10)
        X = from_unit_cube(latin_hypercube(n_init, dim), lb, ub)
        y = np.array([[f(x)] for x in X], dtype=float)
        best_hist = [float(np.min(y))] * len(y)
    else:
        X, y, best_hist = state.X, state.y, state.best_hist
        print(f"[GP+EI] Resume: {os.path.basename(checkpoint)} from iter={len(y)}")

    try:
        while len(y) < iters:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(dim), nu=2.5
            ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2))
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=seed,
                n_restarts_optimizer=1 if dim >= 20 else 2,
            )
            gp.fit(X, y.ravel())

            n_cand = 2000 if dim >= 20 else 4000
            X_cand = rng.uniform(lb, ub, size=(n_cand, dim))
            mu, sigma = gp.predict(X_cand, return_std=True)
            ei = expected_improvement(mu, sigma, best=float(np.min(y)))
            x_next = X_cand[np.argmax(ei)]
            y_next = f(x_next)

            X = np.vstack([X, x_next[None, :]])
            y = np.vstack([y, [[y_next]]])
            best_hist.append(float(np.min(y)))
            save_state(checkpoint, RunState(X=X, y=y, best_hist=best_hist, extra={}))
            print(f"[GP+EI] {os.path.basename(checkpoint)} iter={len(y):03d} best={best_hist[-1]:.6f}")
    except KeyboardInterrupt:
        print(f"\n[GP+EI] Interrupted: {checkpoint}")

    return X, y, best_hist


def benchmark_task(
    func_name: str,
    dim: int,
    func: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    iters: int,
    seed: int,
    ckpt_dir: str,
    resume: bool,
):
    lb = np.full(dim, bounds[0], dtype=float)
    ub = np.full(dim, bounds[1], dtype=float)
    stem = f"{func_name}_d{dim}_s{seed}"
    turbo_ckpt = os.path.join(ckpt_dir, f"{stem}_turbo.pkl")
    gp_ckpt = os.path.join(ckpt_dir, f"{stem}_gpei.pkl")

    X_t, y_t, hist_t = run_turbo(func, lb, ub, iters, seed, turbo_ckpt, resume)
    X_g, y_g, hist_g = run_gp_ei(func, lb, ub, iters, seed, gp_ckpt, resume)
    return hist_t, hist_g, float(np.min(y_t)), float(np.min(y_g))


def plot_grid(results, out_png):
    import matplotlib.pyplot as plt

    funcs = ["ackley", "f4"]
    dims = [10, 20, 50]
    fig, axes = plt.subplots(len(funcs), len(dims), figsize=(15, 8), sharex=False, sharey=False)
    for i, fn in enumerate(funcs):
        for j, d in enumerate(dims):
            ax = axes[i, j]
            r = results[(fn, d)]
            xt = np.arange(1, len(r["turbo_hist"]) + 1)
            xg = np.arange(1, len(r["gp_hist"]) + 1)
            ax.plot(xt, r["turbo_hist"], label="TuRBO(hetero)", linewidth=1.8)
            ax.plot(xg, r["gp_hist"], label="GP+EI", linewidth=1.8)
            ax.set_title(f"{fn.upper()}  dim={d}")
            ax.set_xlabel("iter")
            ax.set_ylabel("best")
            ax.grid(alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[Plot] {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=60, help="每个任务总迭代次数（含初始点）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt-dir", type=str, default="bench_ckpt")
    parser.add_argument("--out", type=str, default="bench_ackley_f4_d10_d20_d50.png")
    args = parser.parse_args()

    tasks = {
        "ackley": (ackley, (-5.0, 5.0)),
        "f4": (f4_rosen_log, (-3.0, 3.0)),
    }
    dims = [10, 20, 50]

    results = {}
    for fn_name, (fn, bnd) in tasks.items():
        for d in dims:
            print("\n" + "=" * 70)
            print(f"Task: {fn_name} | dim={d} | iters={args.iters}")
            hist_t, hist_g, best_t, best_g = benchmark_task(
                func_name=fn_name,
                dim=d,
                func=fn,
                bounds=bnd,
                iters=args.iters,
                seed=args.seed,
                ckpt_dir=args.ckpt_dir,
                resume=args.resume,
            )
            results[(fn_name, d)] = {
                "turbo_hist": hist_t,
                "gp_hist": hist_g,
                "best_turbo": best_t,
                "best_gp": best_g,
            }
            print(f"[Result] {fn_name} d={d}: TuRBO={best_t:.6f}, GP+EI={best_g:.6f}")

    plot_grid(results, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
