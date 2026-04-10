"""
Ackley 对比实验（可中断续跑）
================================

目标：
1) 用少量迭代对比改造后的 TuRBO（hetero scoring）与 GP+EI；
2) 输出直观收敛曲线图；
3) 支持中途中断后继续跑（checkpoint）。

用法示例：
    python ackley_compare.py --iters 40 --seed 42
    python ackley_compare.py --iters 40 --seed 42 --resume
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from turbo1 import Turbo1
from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube


def ackley(x: np.ndarray) -> float:
    """标准 Ackley（全局最优在 x=0，f=0）。"""
    x = np.asarray(x, dtype=float)
    d = x.size
    a, b, c = 20.0, 0.2, 2 * np.pi
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return float(term1 + term2 + a + np.e)


def expected_improvement(mu, sigma, best, xi=0.01):
    """最小化场景下的 EI。"""
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
    best_hist: list
    extra: dict


def load_state(path: str) -> RunState | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_state(path: str, state: RunState):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def run_turbo_ackley(lb, ub, iters=40, seed=42, checkpoint_path="ckpt_turbo.pkl", resume=False):
    dim = len(lb)
    rng = np.random.default_rng(seed)
    state = load_state(checkpoint_path) if resume else None

    turbo = Turbo1(
        f=None,
        lb=lb,
        ub=ub,
        n_init=max(6, 2 * dim),
        max_evals=10_000,
        batch_size=1,
        verbose=False,
        use_hetero_lcb=1,
        hetero_beta0=2.0,
        hetero_beta1=0.8,
        hetero_noise_penalty=0.35,
        hetero_k_neighbors=6,
        use_lcb=0,
        budget=iters,
    )

    if state is None:
        X = np.zeros((0, dim), dtype=float)
        y = np.zeros((0, 1), dtype=float)
        best_hist = []
        hypers = {}
    else:
        X = state.X
        y = state.y
        best_hist = state.best_hist
        hypers = state.extra.get("hypers", {})
        turbo._X = np.array(to_unit_cube(X, lb, ub), copy=True)
        turbo._fX = np.array(y, copy=True)
        turbo.X = np.array(X, copy=True)
        turbo.fX = np.array(y, copy=True)
        turbo.used_budget = len(y)
        print(f"[TuRBO] Resume from {len(y)} iterations.")

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
                    n_training_steps=60,
                    hypers=hypers,
                    used_budget=len(y),
                )
                x_unit = turbo._select_candidates(X_cand, y_cand)[0]

            x = from_unit_cube(x_unit[None, :], lb, ub).reshape(-1)
            fx = ackley(x)

            if len(y) >= turbo.n_init:
                turbo._adjust_length(np.array([[fx]]))

            X = np.vstack([X, x[None, :]])
            y = np.vstack([y, [[fx]]])
            turbo._X = np.vstack([turbo._X, x_unit[None, :]]) if len(turbo._X) else x_unit[None, :]
            turbo._fX = np.vstack([turbo._fX, [[fx]]]) if len(turbo._fX) else np.array([[fx]])
            turbo.X = np.vstack([turbo.X, x[None, :]])
            turbo.fX = np.vstack([turbo.fX, [[fx]]])

            best_hist.append(float(np.min(y)))
            save_state(checkpoint_path, RunState(X=X, y=y, best_hist=best_hist, extra={"hypers": hypers}))
            print(f"[TuRBO] iter={len(y):03d} f={fx:.6f} best={best_hist[-1]:.6f} length={turbo.length:.4f}")

    except KeyboardInterrupt:
        print("\n[TuRBO] Interrupted. Checkpoint saved.")

    return X, y, best_hist


def run_gp_ei_ackley(lb, ub, iters=40, seed=42, checkpoint_path="ckpt_gp_ei.pkl", resume=False):
    dim = len(lb)
    rng = np.random.default_rng(seed)
    state = load_state(checkpoint_path) if resume else None

    if state is None:
        X = from_unit_cube(latin_hypercube(max(6, 2 * dim), dim), lb, ub)
        y = np.array([[ackley(x)] for x in X], dtype=float)
        best_hist = [float(np.min(y))] * len(y)
    else:
        X, y, best_hist = state.X, state.y, state.best_hist
        print(f"[GP+EI] Resume from {len(y)} iterations.")

    try:
        while len(y) < iters:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(
                noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2)
            )
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=2)
            gp.fit(X, y.ravel())

            # 随机候选 + EI 选点（小机器可跑）
            n_cand = 3000
            X_cand = rng.uniform(lb, ub, size=(n_cand, dim))
            mu, sigma = gp.predict(X_cand, return_std=True)
            ei = expected_improvement(mu, sigma, best=float(np.min(y)))
            x_next = X_cand[np.argmax(ei)]
            y_next = ackley(x_next)

            X = np.vstack([X, x_next[None, :]])
            y = np.vstack([y, [[y_next]]])
            best_hist.append(float(np.min(y)))
            save_state(checkpoint_path, RunState(X=X, y=y, best_hist=best_hist, extra={}))
            print(f"[GP+EI] iter={len(y):03d} f={y_next:.6f} best={best_hist[-1]:.6f}")

    except KeyboardInterrupt:
        print("\n[GP+EI] Interrupted. Checkpoint saved.")

    return X, y, best_hist


def plot_results(turbo_hist, gp_hist, out_png="ackley_compare.png"):
    import matplotlib.pyplot as plt

    n = max(len(turbo_hist), len(gp_hist))
    x_t = np.arange(1, len(turbo_hist) + 1)
    x_g = np.arange(1, len(gp_hist) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x_t, turbo_hist, label="TuRBO (hetero)", linewidth=2)
    plt.plot(x_g, gp_hist, label="GP+EI", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far Ackley")
    plt.title("Ackley convergence: TuRBO(hetero) vs GP+EI")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[Plot] Saved -> {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=40, help="总迭代次数（建议 20~60）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 续跑")
    parser.add_argument("--out", type=str, default="ackley_compare.png")
    parser.add_argument("--lb", type=float, default=-5.0)
    parser.add_argument("--ub", type=float, default=5.0)
    args = parser.parse_args()

    lb = np.array([args.lb, args.lb], dtype=float)
    ub = np.array([args.ub, args.ub], dtype=float)

    X_t, y_t, hist_t = run_turbo_ackley(lb, ub, iters=args.iters, seed=args.seed, resume=args.resume)
    X_g, y_g, hist_g = run_gp_ei_ackley(lb, ub, iters=args.iters, seed=args.seed, resume=args.resume)
    plot_results(hist_t, hist_g, out_png=args.out)

    print("\n==== Final result ====")
    print(f"TuRBO best: {float(np.min(y_t)):.6f}")
    print(f"GP+EI best: {float(np.min(y_g)):.6f}")


if __name__ == "__main__":
    main()
