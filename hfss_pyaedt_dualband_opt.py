"""PyAEDT HFSS dual-band simulation + optimization wrapper.

需求覆盖：
1) 打开现有 HFSS 工程并加载设计；
2) 将设计变量封装为函数参数（DesignVariables dataclass）；
3) 调用 Setup/Sweep 运行仿真；
4) 提取 S11 曲线；
5) 检查 26-32 GHz 与 37-39 GHz 内 |S11|<-10 dB；
6) 提取 28/38 GHz 增益（Infinite Sphere1）；
7) 同时接入现有改进优化算法（optimizer.py + turbo1.py）；
8) 返回仿真结果并导出；
9) 包含完整路径占位与异常处理。
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    from ansys.aedt.core import Hfss
except Exception:  # noqa: BLE001
    from pyaedt import Hfss

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


# -----------------------------
# 工程参数（可直接改成你本机绝对路径）
# -----------------------------
PROJECT_NAME = "A1"
DESIGN_NAME = "HFSSDesign1"
SETUP_NAME = "Setup1"
SWEEP_NAME = "Sweep"
FAR_FIELD_SPHERE = "Infinite Sphere1"

# 完整路径占位：请改为你的工程绝对路径，例如 r"D:\\hfss\\A1.aedt"
PROJECT_PATH = r"<FULL_PATH_TO_YOUR_A1.aedt>"
AUTO_REMOVE_LOCK = True

BAND_1 = (26.0, 32.0)
BAND_2 = (37.0, 39.0)
TARGET_FREQS = (28.0, 38.0)
S11_THRESHOLD_DB = -10.0
STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    """捕获 Ctrl+C / 终止信号，安全中断当前批次流程。"""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[INTERRUPT] 收到信号 {signum}，将在当前步骤后安全退出。")


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)


class AnalyzeHeartbeat:
    """analyze_setup 期间打印心跳，避免长时间无输出。"""

    def __init__(self, tag: str, interval_sec: int = 30):
        self.tag = tag
        self.interval_sec = max(5, int(interval_sec))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            elapsed = time.time() - self._start_time
            print(f"[PROGRESS] {self.tag} 进行中... 已用时 {elapsed:.0f}s")

    def __enter__(self):
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        elapsed = time.time() - self._start_time
        print(f"[PROGRESS] {self.tag} 结束，总耗时 {elapsed:.1f}s")


def _request_aedt_stop(hfss: Hfss) -> None:
    """尽力请求 AEDT 停止当前仿真。"""
    stop_calls = [
        ("hfss.stop_simulations", lambda: hfss.stop_simulations()),
        ("hfss.odesktop.StopSimulations", lambda: hfss.odesktop.StopSimulations()),
        ("hfss.desktop_class.odesktop.StopSimulations", lambda: hfss.desktop_class.odesktop.StopSimulations()),
    ]
    for tag, fn in stop_calls:
        try:
            fn()
            print(f"[INTERRUPT] 已发送停止命令: {tag}")
            return
        except Exception:
            continue
    print("[INTERRUPT] 未找到可用停止接口，请等待当前求解步结束。")


def _run_analyze_with_interrupt(hfss: Hfss, setup_name: str) -> None:
    """
    在后台线程执行 analyze_setup，主线程轮询中断并尝试停止 AEDT 求解。
    """
    box: dict[str, Any] = {"exc": None}

    def _target():
        try:
            hfss.analyze_setup(setup_name)
        except Exception as exc:  # noqa: BLE001
            box["exc"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    while thread.is_alive():
        if STOP_REQUESTED:
            _request_aedt_stop(hfss)
            thread.join(timeout=8.0)
            raise KeyboardInterrupt("用户中断：已请求停止 HFSS 仿真。")
        thread.join(timeout=1.0)

    if box["exc"] is not None:
        raise box["exc"]


@dataclass
class DesignVariables:
    """HFSS 参数（单位 mm）。"""

    W: float = 15.0
    h: float = 0.787
    Lx: float = 9.3
    Ly: float = 1.15
    dy: float = 2.27
    dc: float = 1.0
    Rc: float = 6.48
    Sl: float = 3.97
    Sw: float = 0.4
    d: float = 0.8
    y2: float = 1.46
    xx: float = 0.0
    S: float = 1.2
    dp: float = 0.42
    x1: float = 1.8
    y1: float = 1.65


def _ensure_numpy_compat() -> None:
    """兼容部分依赖在 numpy 2.x 下使用旧别名。"""
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "int_"):
        np.int_ = np.int64  # type: ignore[attr-defined]
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_  # type: ignore[attr-defined]
    if not hasattr(np, "bool_"):
        np.bool_ = bool  # type: ignore[attr-defined]


def _ensure_sklearn_compat() -> None:
    """兼容旧优化器对 sklearn.datasets.load_boston 的硬依赖。"""
    try:
        import sklearn.datasets as sk_datasets  # type: ignore
    except Exception:
        return
    if hasattr(sk_datasets, "load_boston"):
        return

    def _fake_load_boston(*_args, **_kwargs):
        raise RuntimeError("load_boston is unavailable; compatibility shim injected by hfss_pyaedt_dualband_opt.py")

    sk_datasets.load_boston = _fake_load_boston  # type: ignore[attr-defined]


class FallbackRandomOptimizer:
    """当 bayesmark/optimizer 导入失败时的兜底优化器。"""

    def __init__(self, api_config: dict[str, dict[str, Any]], seed: int = 42):
        self.api_config = api_config
        self.rng = np.random.default_rng(seed)

    def suggest(self, n_suggestions: int = 1) -> list[dict[str, float]]:
        points: list[dict[str, float]] = []
        for _ in range(n_suggestions):
            one: dict[str, float] = {}
            for key, cfg in self.api_config.items():
                lo, hi = cfg["range"]
                one[key] = float(self.rng.uniform(lo, hi))
            points.append(one)
        return points

    def observe(self, X_observed: list[dict[str, float]], Y_observed: list[float]) -> None:
        _ = (X_observed, Y_observed)


def _patch_optimizer_config(optimizer_cls: type) -> None:
    """注入 turbo1.py 中改进过的异质方差参数。"""

    original = optimizer_cls._read_config

    def patched(self):
        cfg = original(self)
        cfg["turbo_training_steps"] = 80
        cfg["reset_no_improvement"] = 1_000_000
        cfg["turbo"]["use_hetero_lcb"] = 1
        cfg["turbo"]["hetero_beta0"] = 2.2
        cfg["turbo"]["hetero_beta1"] = 0.8
        cfg["turbo"]["hetero_noise_penalty"] = 0.35
        cfg["turbo"]["hetero_k_neighbors"] = 6
        return cfg

    optimizer_cls._read_config = patched


def build_optimizer(api_config: dict[str, dict[str, Any]]):
    """优先加载当前仓库 optimizer.py（含 turbo1.py）。"""
    _ensure_numpy_compat()
    _ensure_sklearn_compat()
    try:
        import optimizer as local_optimizer

        if hasattr(local_optimizer, "DEBUG"):
            local_optimizer.DEBUG = False

        optimizer_cls = local_optimizer.SpacePartitioningOptimizer
        _patch_optimizer_config(optimizer_cls)
        return optimizer_cls(api_config=api_config)
    except Exception as exc:
        print(f"[WARN] 使用 SpacePartitioningOptimizer 失败，退化随机优化器: {exc}")
        return FallbackRandomOptimizer(api_config)


def _apply_design_variables(hfss: Hfss, variables: DesignVariables) -> None:
    for name, value in asdict(variables).items():
        hfss[name] = f"{value:.6f}mm"


def _possible_lock_files(project_file: Path) -> list[Path]:
    stem = project_file.stem
    parent = project_file.parent
    return [
        Path(str(project_file) + ".lock"),
        parent / f"{stem}.lock",
    ]


def _remove_project_lock(project_file: Path) -> bool:
    removed = False
    for lock_file in _possible_lock_files(project_file):
        if lock_file.exists():
            try:
                lock_file.unlink()
                print(f"[LOCK] 已移除锁文件: {lock_file}")
                removed = True
            except Exception as exc:  # noqa: BLE001
                print(f"[LOCK] 锁文件删除失败: {lock_file}, err={exc}")
    return removed


def _attach_existing_hfss(project_file: Path, non_graphical: bool, version: str):
    """
    当 Hfss(...) 构造触发 `__init__ should return None, not 'bool'` 时，
    尝试附着到已打开的 Desktop + Project + Design。
    """
    project_name = project_file.stem
    candidates: list[tuple[str, Any]] = []
    try:
        from ansys.aedt.core import Desktop, get_pyaedt_app  # type: ignore

        candidates.append(("ansys.aedt.core", (Desktop, get_pyaedt_app)))
    except Exception:
        pass
    try:
        from pyaedt import Desktop, get_pyaedt_app  # type: ignore

        candidates.append(("pyaedt", (Desktop, get_pyaedt_app)))
    except Exception:
        pass

    attach_errors: list[str] = []
    for source, (Desktop, get_pyaedt_app) in candidates:
        try:
            # 先连接/复用已有 Desktop 会话。
            try:
                Desktop(new_desktop=False, non_graphical=non_graphical, version=version)
            except TypeError:
                Desktop(new_desktop_session=False, non_graphical=non_graphical, specified_version=version)

            # 再尝试附着应用对象（多种签名都试）。
            call_variants = [
                {"project_name": project_name, "design_name": DESIGN_NAME},
                {"project": project_name, "design": DESIGN_NAME},
                {"project_name": str(project_file), "design_name": DESIGN_NAME},
                {"project": str(project_file), "design": DESIGN_NAME},
                {"projectname": project_name, "designname": DESIGN_NAME},
                {},
            ]
            for kwargs in call_variants:
                try:
                    app = get_pyaedt_app(**kwargs) if kwargs else get_pyaedt_app()
                    if app is not None:
                        if hasattr(app, "set_active_design"):
                            try:
                                app.set_active_design(DESIGN_NAME)
                            except Exception:
                                pass
                        return app
                except Exception as e:  # noqa: BLE001
                    attach_errors.append(f"{source}.get_pyaedt_app({kwargs}): {e}")
        except Exception as e:  # noqa: BLE001
            attach_errors.append(f"{source}.Desktop attach failed: {e}")

    raise RuntimeError("附着已打开 HFSS 会话失败: " + " | ".join(attach_errors))


def _create_hfss_session(project_file: Path, non_graphical: bool, version: str):
    """
    兼容不同 PyAEDT 版本的 Hfss 构造参数。

    说明：
    - 新版常见参数：project/design/version
    - 旧版常见参数：projectname/designname/specified_version
    """
    attempts: list[tuple[str, dict[str, Any]]] = [
        (
            "new_signature",
            {
                "project": str(project_file),
                "design": DESIGN_NAME,
                "version": version,
                "non_graphical": non_graphical,
                "new_desktop": False,
                "remove_lock": False,
            },
        ),
        (
            "legacy_signature",
            {
                "projectname": str(project_file),
                "designname": DESIGN_NAME,
                "specified_version": version,
                "non_graphical": non_graphical,
                "new_desktop": False,
                "remove_lock": False,
            },
        ),
        (
            "legacy_without_version",
            {
                "projectname": str(project_file),
                "designname": DESIGN_NAME,
                "non_graphical": non_graphical,
                "new_desktop": False,
                "remove_lock": False,
            },
        ),
        (
            "new_signature_remove_lock",
            {
                "project": str(project_file),
                "design": DESIGN_NAME,
                "version": version,
                "non_graphical": non_graphical,
                "new_desktop": False,
                "remove_lock": True,
            },
        ),
    ]

    errors: list[str] = []
    saw_lock_error = False
    for tag, kwargs in attempts:
        try:
            return Hfss(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if "Project is locked" in str(exc):
                saw_lock_error = True
            errors.append(f"{tag}: {exc}")

    if saw_lock_error and AUTO_REMOVE_LOCK and _remove_project_lock(project_file):
        print("[LOCK] 检测到项目锁，移除后重试 HFSS 连接。")
        retry_kwargs = {
            "project": str(project_file),
            "design": DESIGN_NAME,
            "version": version,
            "non_graphical": non_graphical,
            "new_desktop": False,
            "remove_lock": True,
        }
        try:
            return Hfss(**retry_kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"retry_after_remove_lock: {exc}")

    # 某些 PyAEDT 版本在 __init__ 内部异常时会返回 False，触发该 TypeError。
    # 这时尝试“附着到现有 Desktop 会话”作为兜底。
    if all("__init__() should return None, not 'bool'" in e for e in errors):
        try:
            return _attach_existing_hfss(project_file, non_graphical=non_graphical, version=version)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attach_existing: {exc}")

    joined = " | ".join(errors)
    raise RuntimeError(
        "无法初始化 Hfss 会话。请检查 PyAEDT 版本与参数签名。"
        f" 已尝试: {joined}"
    )


def _get_s11_curve(hfss: Hfss) -> tuple[np.ndarray, np.ndarray]:
    setup_candidates = [
        f"{SETUP_NAME} : {SWEEP_NAME}",
        f"{SETUP_NAME}:{SWEEP_NAME}",
        f"{SETUP_NAME} : LastAdaptive",
        f"{SETUP_NAME}:LastAdaptive",
        SETUP_NAME,
    ]
    expression_candidates = ["dB(S(1,1))", "S(1,1)"]
    last_err: Exception | None = None

    for setup in setup_candidates:
        for expr in expression_candidates:
            try:
                sol = hfss.post.get_solution_data(expressions=expr, setup_sweep_name=setup)
                if sol is None or isinstance(sol, bool) or not hasattr(sol, "primary_sweep_values"):
                    continue
                freqs = np.array(sol.primary_sweep_values, dtype=float)
                vals = np.array(sol.data_real(), dtype=float)
                if freqs.size == 0 or vals.size == 0:
                    continue
                if expr == "S(1,1)":
                    vals = 20.0 * np.log10(np.maximum(np.abs(vals), 1e-15))
                return freqs, vals
            except Exception as exc:  # noqa: BLE001
                last_err = exc

    raise RuntimeError(f"未提取到 S11 曲线，请检查 setup/sweep/expression。last={last_err}")


def _band_ok(freqs: np.ndarray, s11_db: np.ndarray, band: tuple[float, float], threshold: float) -> bool:
    f_lo, f_hi = band
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return bool(np.any(mask) and np.all(s11_db[mask] < threshold))


def _extract_gain_db(hfss: Hfss, target_freq_ghz: float) -> float:
    setup_candidates = [
        f"{SETUP_NAME} : {SWEEP_NAME}",
        f"{SETUP_NAME}:{SWEEP_NAME}",
        f"{SETUP_NAME} : LastAdaptive",
        f"{SETUP_NAME}:LastAdaptive",
        SETUP_NAME,
    ]
    # 避免触发历史工程里的 `gaintotal` 宏报错，仅使用 RealizedGainTotal。
    expression_candidates = ["dB(RealizedGainTotal)", "RealizedGainTotal"]
    context_candidates = [FAR_FIELD_SPHERE, "Infinite Sphere1", None]
    fail_reasons: list[str] = []
    freq_tag = f"{target_freq_ghz}GHz"

    query_variants = [
        {"domain": "Sweep", "variations": {"Freq": [freq_tag], "Theta": ["All"], "Phi": ["All"]}},
        {"variations": {"Freq": [freq_tag], "Theta": ["All"], "Phi": ["All"]}},
        {"intrinsics": {"Freq": freq_tag, "Theta": "All", "Phi": "All"}},
        {},
    ]

    for setup in setup_candidates:
        for expr in expression_candidates:
            for context in context_candidates:
                for qv in query_variants:
                    try:
                        kwargs = {"expressions": expr, "setup_sweep_name": setup}
                        if context is not None:
                            kwargs["context"] = context
                        kwargs.update(qv)

                        sol = hfss.post.get_solution_data(**kwargs)
                        if sol is None or isinstance(sol, bool):
                            fail_reasons.append(f"{setup}|{expr}|{context}|{qv}: empty-solution({type(sol).__name__})")
                            continue

                        vals = np.array(sol.data_real(), dtype=float)
                        if vals.size == 0:
                            fail_reasons.append(f"{setup}|{expr}|{context}|{qv}: empty-values")
                            continue

                        if expr == "RealizedGainTotal":
                            vals = 10.0 * np.log10(np.maximum(np.abs(vals), 1e-15))

                        # Far-field 数据主扫常常是 Theta/Phi；此时直接取该频点上的峰值增益。
                        return float(np.nanmax(vals))
                    except Exception as exc:  # noqa: BLE001
                        fail_reasons.append(f"{setup}|{expr}|{context}|{qv}: {exc}")

    tail = " | ".join(fail_reasons[-5:]) if fail_reasons else "unknown"
    print(f"[WARN] 提取 {target_freq_ghz} GHz 增益失败，返回 NaN。recent={tail}")
    return float("nan")


def _evaluate_with_open_hfss(hfss: Hfss, design_vars: DesignVariables, project_path: str) -> dict[str, Any]:
    """在已打开的 HFSS 会话中评估一次设计。"""
    _apply_design_variables(hfss, design_vars)
    if STOP_REQUESTED:
        raise KeyboardInterrupt("检测到用户中断请求，停止 analyze_setup。")
    with AnalyzeHeartbeat(tag=f"HFSS analyze_setup({SETUP_NAME})", interval_sec=30):
        _run_analyze_with_interrupt(hfss, SETUP_NAME)
    if STOP_REQUESTED:
        raise KeyboardInterrupt("检测到用户中断请求，停止后处理提取。")

    freqs, s11_db = _get_s11_curve(hfss)
    band1_ok = _band_ok(freqs, s11_db, BAND_1, S11_THRESHOLD_DB)
    band2_ok = _band_ok(freqs, s11_db, BAND_2, S11_THRESHOLD_DB)

    gain_28 = _extract_gain_db(hfss, TARGET_FREQS[0])
    gain_38 = _extract_gain_db(hfss, TARGET_FREQS[1])

    return {
        "project_name": PROJECT_NAME,
        "design_name": DESIGN_NAME,
        "setup_name": SETUP_NAME,
        "sweep_name": SWEEP_NAME,
        "far_field_sphere": FAR_FIELD_SPHERE,
        "project_path": project_path,
        "design_vars": asdict(design_vars),
        "s11_curve": {
            "freq_ghz": freqs.tolist(),
            "s11_db": s11_db.tolist(),
        },
        "gain_28ghz_db": gain_28,
        "gain_38ghz_db": gain_38,
        "band_26_32_ok": band1_ok,
        "band_37_39_ok": band2_ok,
        "dualband_match_ok": bool(band1_ok and band2_ok),
    }


def run_single_simulation(
    project_path: str,
    design_vars: DesignVariables,
    setup_name: str = SETUP_NAME,
    sweep_name: str = SWEEP_NAME,
    non_graphical: bool = True,
    version: str = "2025.1",
) -> dict[str, Any]:
    """运行一次仿真并返回结果字典。"""
    _ = (setup_name, sweep_name)  # 预留参数，当前使用全局 setup/sweep 常量

    if STOP_REQUESTED:
        raise KeyboardInterrupt("检测到用户中断请求，跳过本次仿真。")

    project_file = Path(project_path)
    if not project_file.exists():
        raise FileNotFoundError(f"HFSS 工程不存在: {project_file}")

    hfss = None
    try:
        hfss = _create_hfss_session(project_file, non_graphical=non_graphical, version=version)
        return _evaluate_with_open_hfss(hfss, design_vars, str(project_file))
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeError(f"仿真失败: {exc}") from exc
    finally:
        if hfss is not None:
            try:
                hfss.release_desktop(close_projects=False, close_desktop=False)
            except TypeError:
                hfss.release_desktop()
            except Exception:
                pass


def _objective(result: dict[str, Any]) -> float:
    """最小化目标：频段约束惩罚 + 负增益 + S11 最优值。"""
    freqs = np.asarray(result["s11_curve"]["freq_ghz"], dtype=float)
    s11 = np.asarray(result["s11_curve"]["s11_db"], dtype=float)
    g28 = float(result["gain_28ghz_db"])
    g38 = float(result["gain_38ghz_db"])

    penalty = 0.0
    for ok_key, band in (("band_26_32_ok", BAND_1), ("band_37_39_ok", BAND_2)):
        if not result.get(ok_key, False):
            mask = (freqs >= band[0]) & (freqs <= band[1])
            if np.any(mask):
                penalty += float(np.mean(np.maximum(s11[mask] - S11_THRESHOLD_DB, 0.0))) * 20.0
            else:
                penalty += 500.0

    gain_term = 0.0
    if math.isfinite(g28):
        gain_term -= 0.5 * g28
    if math.isfinite(g38):
        gain_term -= 0.5 * g38

    s11_term = float(np.min(s11)) if s11.size else 100.0
    return penalty + gain_term + s11_term


def _export_s11_csv(result: dict[str, Any], output_csv: str) -> None:
    freqs = result["s11_curve"]["freq_ghz"]
    s11 = result["s11_curve"]["s11_db"]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["freq_ghz", "s11_db"])
        for fr, v in zip(freqs, s11):
            writer.writerow([fr, v])


def run_optimization(
    project_path: str,
    budget: int = 20,
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """调用 optimizer.py + turbo1.py 进行参数优化，并导出结果。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    api_config = {
        "Rc": {"type": "real", "space": "linear", "range": (5.5, 7.5)},
        "S": {"type": "real", "space": "linear", "range": (0.6, 1.8)},
        "dp": {"type": "real", "space": "linear", "range": (0.2, 0.7)},
        "x1": {"type": "real", "space": "linear", "range": (1.0, 2.5)},
        "y1": {"type": "real", "space": "linear", "range": (1.0, 2.2)},
    }
    optimizer = build_optimizer(api_config)

    iter_csv = Path(output_dir) / "optimization_log.csv"
    best_json = Path(output_dir) / "best_result.json"
    best_s11_csv = Path(output_dir) / "best_s11_curve.csv"

    best_result: dict[str, Any] | None = None
    best_loss = float("inf")
    default_vars = DesignVariables()

    hfss = None
    project_file = Path(project_path)
    if not project_file.exists():
        raise FileNotFoundError(f"HFSS 工程不存在: {project_file}")

    try:
        hfss = _create_hfss_session(project_file, non_graphical=True, version="2025.1")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"优化前 HFSS 会话初始化失败: {exc}") from exc

    try:
        with open(iter_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iter",
                    "loss",
                    "gain_28ghz_db",
                    "gain_38ghz_db",
                    "dualband_match_ok",
                    "design_vars_json",
                    "error",
                ]
            )

            index_iter = range(1, budget + 1)
            if tqdm is not None:
                index_iter = tqdm(
                    index_iter,
                    total=budget,
                    desc="HFSS Optimization",
                    unit="iter",
                    dynamic_ncols=True,
                    leave=True,
                )
            else:
                print("[PROGRESS] 未安装 tqdm，使用普通文本进度输出。可执行: pip install tqdm")

            for i in index_iter:
                if STOP_REQUESTED:
                    print("[INTERRUPT] 检测到中断请求，提前结束优化循环。")
                    break
                cand = optimizer.suggest(1)[0]
                vars_i = DesignVariables(**{**asdict(default_vars), **cand})

                err_msg = ""
                try:
                    result = _evaluate_with_open_hfss(hfss, vars_i, str(project_file))
                    loss = _objective(result)
                except KeyboardInterrupt:
                    print("[INTERRUPT] 单次仿真被用户中断，结束优化并保留已有结果。")
                    break
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "design_vars": asdict(vars_i),
                        "s11_curve": {"freq_ghz": [], "s11_db": []},
                        "gain_28ghz_db": float("nan"),
                        "gain_38ghz_db": float("nan"),
                        "dualband_match_ok": False,
                    }
                    loss = 1e6
                    err_msg = str(exc)

                optimizer.observe([cand], [loss])

                writer.writerow(
                    [
                        i,
                        loss,
                        result.get("gain_28ghz_db", float("nan")),
                        result.get("gain_38ghz_db", float("nan")),
                        int(bool(result.get("dualband_match_ok", False))),
                        json.dumps(result.get("design_vars", {}), ensure_ascii=False),
                        err_msg,
                    ]
                )

                if not err_msg and loss < best_loss:
                    best_loss = loss
                    best_result = result

                print(
                    f"[{i}/{budget}] loss={loss:.4f} dualband={result.get('dualband_match_ok', False)} "
                    f"g28={result.get('gain_28ghz_db')} g38={result.get('gain_38ghz_db')}"
                )
                if tqdm is not None and hasattr(index_iter, "set_postfix"):
                    index_iter.set_postfix(
                        loss=f"{loss:.3f}",
                        dual=bool(result.get("dualband_match_ok", False)),
                    )

        if best_result is None:
            final = {
                "message": "没有成功仿真样本",
                "project_path": project_path,
                "optimizer": type(optimizer).__name__,
            }
        else:
            final = {
                "best_loss": best_loss,
                "optimizer": type(optimizer).__name__,
                **best_result,
            }
            _export_s11_csv(final, str(best_s11_csv))

        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
    finally:
        if hfss is not None:
            try:
                hfss.release_desktop(close_projects=False, close_desktop=False)
            except TypeError:
                hfss.release_desktop()
            except Exception:
                pass

    return {
        "best_result_json": str(best_json),
        "best_s11_csv": str(best_s11_csv) if best_result is not None else "",
        "iter_log_csv": str(iter_csv),
        "summary": final,
    }


def main() -> None:
    try:
        # 1) 单次仿真（用默认参数）
        one_shot = run_single_simulation(
            project_path=PROJECT_PATH,
            design_vars=DesignVariables(),
            setup_name=SETUP_NAME,
            sweep_name=SWEEP_NAME,
        )

        Path("outputs").mkdir(exist_ok=True)
        with open("outputs/single_run_result.json", "w", encoding="utf-8") as f:
            json.dump(one_shot, f, ensure_ascii=False, indent=2)
        _export_s11_csv(one_shot, "outputs/single_run_s11.csv")

        print("\n=== 单次仿真结果 ===")
        print(
            json.dumps(
                {
                    "gain_28ghz_db": one_shot["gain_28ghz_db"],
                    "gain_38ghz_db": one_shot["gain_38ghz_db"],
                    "dualband_match_ok": one_shot["dualband_match_ok"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        if STOP_REQUESTED:
            print("[INTERRUPT] 用户中断后仅完成单次仿真，跳过优化。")
            return

        # 2) 优化（调用改进算法）
        opt_result = run_optimization(project_path=PROJECT_PATH, budget=20, output_dir="outputs")

        print("\n=== 优化结果文件 ===")
        print(json.dumps(opt_result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 用户主动中断，脚本已安全退出。")


if __name__ == "__main__":
    main()
