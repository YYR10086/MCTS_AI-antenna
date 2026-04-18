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
import glob
import hashlib
import json
import logging
import math
import os
import platform
import signal
import subprocess
import threading
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
try:
    from ansys.aedt.core import Hfss
except Exception:  # noqa: BLE001
    from pyaedt import Hfss

# -----------------------------
# 工程参数（可直接改成你本机绝对路径）
# -----------------------------
PROJECT_NAME = "A1"
DESIGN_NAME = "HFSSDesign1"
SETUP_NAME = "Setup1"
SWEEP_NAME = "Sweep"
FAR_FIELD_SPHERE = "Infinite Sphere1"

# 默认读取脚本同目录下的 A1.aedt
PROJECT_PATH = str(Path(__file__).parent / "A1.aedt")
AUTO_REMOVE_LOCK = True

BAND_1 = (26.0, 32.0)
BAND_2 = (37.0, 39.0)
TARGET_FREQS = (28.0, 38.0)
S11_THRESHOLD_DB = -10.0
STOP_REQUESTED = False
GAIN_PHYSICAL_MAX = 30.0
DEDUP_THRESHOLD = 1e-4  # 仅将“数值误差级别”的参数差异视为重复
SESSION_REFRESH_INTERVAL = 10  # 每 N 轮主动重建一次 HFSS 会话
BUDGET = 100  # 服务器上一次跑100轮
_DIAG_PRINTED = False

API_CONFIG = {
    "W": {"type": "real", "space": "linear", "range": (12.0, 20.0)},
    "Lx": {"type": "real", "space": "linear", "range": (8.0, 10.0)},
    "Ly": {"type": "real", "space": "linear", "range": (0.5, 2.0)},
    "dy": {"type": "real", "space": "linear", "range": (0.5, 4.0)},
    "Rc": {"type": "real", "space": "linear", "range": (5.0, 7.0)},
    "dc": {"type": "real", "space": "linear", "range": (0.5, 1.3)},
    "S": {"type": "real", "space": "linear", "range": (1.0, 1.5)},
    "d": {"type": "real", "space": "linear", "range": (0.5, 1.0)},
    "Sw": {"type": "real", "space": "linear", "range": (0.1, 1.0)},
    "dp": {"type": "real", "space": "linear", "range": (0.1, 1.0)},
    "Sl": {"type": "real", "space": "linear", "range": (1.0, 5.0)},
    "x1": {"type": "real", "space": "linear", "range": (1.0, 5.0)},
    "y1": {"type": "real", "space": "linear", "range": (0.0, 3.0)},
    "y2": {"type": "real", "space": "linear", "range": (0.0, 3.0)},
}
OPT_PARAM_NAMES = [
    "W",
    "Lx",
    "Ly",
    "dy",
    "Rc",
    "dc",
    "S",
    "d",
    "Sw",
    "dp",
    "Sl",
    "x1",
    "y1",
    "y2",
]
# 以下两个变量固定不优化，不写入此列表：
# h  = 0.787mm（基板厚度，固定）
# xx = 0mm（固定为0）
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _request_stop(signum, _frame) -> None:
    """捕获 Ctrl+C / 终止信号，安全中断当前批次流程。"""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[INTERRUPT] 收到信号 {signum}，将在当前步骤后安全退出。")


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)


def _kill_stale_aedt() -> None:
    """强制结束所有残留的 AEDT 进程（仅 Windows）。"""
    if platform.system() != "Windows":
        return
    targets = ["ansysedt.exe", "ansysedtsv.exe", "ANSYSEDT.exe"]
    for name in targets:
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[CLEANUP] 已结束残留进程: {name}")
        except Exception:
            pass
    time.sleep(3)


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
            try:
                hfss.analyze_setup(setup_name)
            except TypeError:
                hfss.odesign.Analyze(setup_name)
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
        print("[DIAG] Setup求解失败，尝试获取HFSS消息日志...")
        try:
            msgs = hfss.odesktop.GetMessages("", "", 2)
            for m in (msgs or [])[-10:]:
                print(f"[HFSS MSG] {m}")
        except Exception:
            pass
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
    except Exception as exc:
        print(f"[WARN] sklearn 不可用，跳过 load_boston 兼容补丁: {exc}")
        return

    # 不能用 hasattr/getattr 访问 load_boston；在 sklearn>=1.2 会直接抛 ImportError。
    if "load_boston" in vars(sk_datasets):
        return

    def _fake_load_boston(*_args, **_kwargs):
        raise RuntimeError("load_boston is unavailable; compatibility shim injected by hfss_pyaedt_dualband_opt.py")

    try:
        setattr(sk_datasets, "load_boston", _fake_load_boston)
    except Exception as exc:
        print(f"[WARN] 无法注入 load_boston 兼容符号，优化器将走兜底路径: {exc}")


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


def validate_params(params: dict) -> dict:
    """参数越界时自动裁剪到 API_CONFIG 合法范围，并记录 warning。"""
    fixed = dict(params)
    for k, cfg in API_CONFIG.items():
        if k in fixed:
            lb, ub = cfg["range"]
            val = float(fixed[k])
            clipped = float(np.clip(val, lb, ub))
            if abs(clipped - val) > 1e-9:
                logging.warning("参数 %s=%.4f 超界，clip到 [%.4f, %.4f]", k, val, lb, ub)
            fixed[k] = clipped
    return fixed


def _apply_design_variables(hfss, design_vars: dict):
    global _DIAG_PRINTED
    if not _DIAG_PRINTED:
        logging.info("hfss 对象类型: %s", type(hfss))
        logging.info("hfss 有 _odesign: %s", hasattr(hfss, "_odesign"))
        logging.info("hfss 有 odesign: %s", hasattr(hfss, "odesign"))
        logging.info("hfss 有 variable_manager: %s", hasattr(hfss, "variable_manager"))
        _DIAG_PRINTED = True

    for name, value in design_vars.items():
        expr = f"{float(value):.6f}mm"
        success = False

        # 方式0：PyWin32 COM 接口（HFSS 2020 R1 专用）
        if not success:
            try:
                odesign = (
                    hfss._odesign
                    if hasattr(hfss, "_odesign") and hfss._odesign is not None
                    else hfss.odesign
                    if hasattr(hfss, "odesign")
                    else None
                )
                if odesign is not None:
                    odesign.ChangeProperty(
                        [
                            "NAME:AllTabs",
                            [
                                "NAME:LocalVariableTab",
                                ["NAME:PropServers", "LocalVariables"],
                                ["NAME:ChangedProps", ["NAME:" + name, "Value:=", expr]],
                            ],
                        ]
                    )
                    success = True
            except Exception as e:
                logging.warning("方式0写入 %s 失败: %s", name, e)

        # 方式0b：直接用 SetVariableValue（PyWin32 旧版接口）
        if not success:
            try:
                odesign = (
                    hfss._odesign
                    if hasattr(hfss, "_odesign") and hfss._odesign is not None
                    else hfss.odesign
                    if hasattr(hfss, "odesign")
                    else None
                )
                if odesign is not None:
                    odesign.SetVariableValue(name, expr)
                    success = True
            except Exception as e:
                logging.warning("方式0b写入 %s 失败: %s", name, e)

        # 方式1a：旧版 PyAEDT COM 接口（odesign，无下划线）
        if not success:
            try:
                hfss.odesign.ChangeProperty(
                    [
                        "NAME:AllTabs",
                        [
                            "NAME:LocalVariableTab",
                            ["NAME:PropServers", "LocalVariables"],
                            ["NAME:ChangedProps", ["NAME:" + name, "Value:=", expr]],
                        ],
                    ]
                )
                success = True
            except Exception as e:
                logging.warning("方式1a写入 %s 失败: %s", name, e)

        # 方式1b：新版 PyAEDT gRPC 接口（_odesign，有下划线）
        if not success:
            try:
                hfss._odesign.ChangeProperty(
                    [
                        "NAME:AllTabs",
                        [
                            "NAME:LocalVariableTab",
                            ["NAME:PropServers", "LocalVariables"],
                            ["NAME:ChangedProps", ["NAME:" + name, "Value:=", expr]],
                        ],
                    ]
                )
                success = True
            except Exception as e:
                logging.warning("方式1b写入 %s 失败: %s", name, e)

        # 方式2：variable_manager 字典接口
        if not success:
            try:
                hfss.variable_manager[name] = expr
                success = True
            except Exception as e:
                logging.warning("方式2写入 %s 失败: %s", name, e)

        # 方式3：hfss[] 下标接口
        if not success:
            try:
                hfss[name] = expr
                success = True
            except Exception as e:
                logging.warning("方式3写入 %s 失败: %s", name, e)

        if not success:
            raise RuntimeError(f"变量 '{name}' 六种方式均写入失败")


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
            Desktop(new_desktop=False, non_graphical=non_graphical, version=version)

            # 再尝试附着应用对象（多种签名都试）。
            call_variants = [
                {"project_name": project_name, "design_name": DESIGN_NAME},
                {"project": project_name, "design": DESIGN_NAME},
                {"project_name": str(project_file), "design_name": DESIGN_NAME},
                {"project": str(project_file), "design": DESIGN_NAME},
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


def _create_hfss_session(project_file, non_graphical=True, version=None):
    lock_file = Path(str(project_file) + ".lock")
    if lock_file.exists():
        try:
            lock_file.unlink()
            logging.info("[LOCK] 已删除锁文件: %s", lock_file)
            time.sleep(2)
        except Exception as e:
            logging.warning("[LOCK] 删除锁文件失败: %s", e)

    versions_to_try = ["2020.1", "2021.1", "2021.2", "2022.1", None]
    if version is not None:
        versions_to_try = [version] + [v for v in versions_to_try if v != version]

    last_exc = None
    for ver in versions_to_try:
        try:
            kwargs = dict(
                project=str(project_file),
                design=DESIGN_NAME,
                non_graphical=non_graphical,
            )
            if ver is not None:
                kwargs["version"] = ver
            hfss = Hfss(**kwargs)
            logging.info("HFSS 会话创建成功，版本: %s", ver or "自动检测")
            return hfss
        except Exception as e:
            logging.warning("版本 %s 尝试失败: %s", ver, e)
            last_exc = e
            time.sleep(2)

    raise RuntimeError(f"所有版本均无法初始化 Hfss 会话: {last_exc}") from last_exc


def _safe_save(hfss: Any) -> None:
    try:
        if hfss is not None and hasattr(hfss, "_oproject") and hfss._oproject is not None:
            hfss.save_project()
    except Exception as e:  # noqa: BLE001
        logging.warning("save_project 失败: %s", e)


def _safe_release(hfss):
    if hfss is None:
        return
    try:
        hfss.release_desktop()
    except Exception as e:
        logging.warning("release_desktop 失败（忽略）: %s", e)


def _cleanup_hfss_session(hfss: Any, sleep_sec: int = 2) -> None:
    if hfss is None:
        return
    _safe_save(hfss)
    _safe_release(hfss)
    time.sleep(sleep_sec)


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


def _build_farfield_variations(hfss: Hfss, freq_ghz: float) -> dict[str, list[str]]:
    """构建 PyAEDT 0.17.5 可接受的 variations（不用 intrinsics）。"""
    variations: dict[str, list[str]] = {}
    try:
        nominal = hfss.available_variations.nominal_values.copy()
        for key, value in nominal.items():
            if isinstance(value, list):
                variations[key] = value
            else:
                variations[key] = [str(value)]
    except Exception:
        pass

    variations["Freq"] = [f"{freq_ghz}GHz"]
    variations["Theta"] = ["All"]
    variations["Phi"] = ["All"]
    return variations


def _peak_from_solution(sol: Any, math_formula: str | None = None) -> float | None:
    if sol is None or isinstance(sol, bool):
        return None
    try:
        vals = np.asarray(sol.data_real(), dtype=float)
    except Exception:
        return None
    if vals.size == 0:
        return None
    if math_formula == "dB":
        vals = 10.0 * np.log10(np.maximum(np.abs(vals), 1e-15))
    return float(np.nanmax(vals))


def _check_geometry_valid(hfss: Hfss) -> bool:
    try:
        _ = hfss.odesign.GetChildObject("Model").GetChildNames()
        return True
    except Exception:
        return False


def _extract_gain_db_once(hfss: Hfss, target_freq_ghz: float) -> float:
    preferred_setup = f"{SETUP_NAME} : {SWEEP_NAME}"
    setup_candidates = [preferred_setup, f"{SETUP_NAME}:{SWEEP_NAME}", f"{SETUP_NAME} : LastAdaptive", SETUP_NAME]
    context = FAR_FIELD_SPHERE
    fail_reasons: list[str] = []

    # C) 先做可用量探测
    report_types: list[str] = []
    ff_quantities: list[str] = []
    try:
        report_types = list(getattr(hfss.post, "available_report_types", []) or [])
    except Exception:
        pass
    print(f"[DEBUG] available report types: {report_types}")

    for setup in setup_candidates:
        try:
            q = hfss.post.available_report_quantities(
                report_category="Far Fields",
                context=context,
            )
            if q:
                ff_quantities = list(q)
                print(f"[DEBUG] far-field quantities @ {setup}: {ff_quantities}")
                break
        except Exception as exc:  # noqa: BLE001
            fail_reasons.append(f"probe_quantities@{setup}: {exc}")

    preferred_quantities = ["RealizedGainTotal", "GainTotal", "dB(RealizedGainTotal)", "dB(GainTotal)"]
    expr_candidates = [q for q in preferred_quantities if q in ff_quantities] or preferred_quantities

    for setup in setup_candidates:
        variations = _build_farfield_variations(hfss, target_freq_ghz)
        primary = "Theta" if "Theta" in variations else ("Phi" if "Phi" in variations else None)

        for expr in expr_candidates:
            for math_formula in (None, "dB"):
                if expr.startswith("dB(") and math_formula == "dB":
                    continue
                try:
                    kwargs = {
                        "expressions": expr,
                        "setup_sweep_name": setup,
                        "domain": "Sweep",
                        "variations": variations,
                        "report_category": "Far Fields",
                        "context": context,
                    }
                    if primary is not None:
                        kwargs["primary_sweep_variable"] = primary
                    if math_formula is not None:
                        kwargs["math_formula"] = math_formula

                    sol = hfss.post.get_solution_data(**kwargs)
                    peak = _peak_from_solution(sol)
                    if peak is not None:
                        return peak
                    fail_reasons.append(f"get_solution_data@{setup}|{expr}|{math_formula}: empty")
                except Exception as exc:  # noqa: BLE001
                    fail_reasons.append(f"get_solution_data@{setup}|{expr}|{math_formula}: {exc}")

            # D fallback 1: get_solution_data_per_variation
            try:
                kwargs2 = {
                    "expression": expr,
                    "setup_sweep_name": setup,
                    "domain": "Sweep",
                    "variations": variations,
                    "report_category": "Far Fields",
                    "context": context,
                }
                if primary is not None:
                    kwargs2["primary_sweep_variable"] = primary
                per_var = hfss.post.get_solution_data_per_variation(**kwargs2)
                if isinstance(per_var, dict):
                    for obj in per_var.values():
                        peak = _peak_from_solution(obj)
                        if peak is not None:
                            return peak
                else:
                    peak = _peak_from_solution(per_var)
                    if peak is not None:
                        return peak
                fail_reasons.append(f"per_variation@{setup}|{expr}: empty")
            except Exception as exc:  # noqa: BLE001
                fail_reasons.append(f"per_variation@{setup}|{expr}: {exc}")

    # D fallback 2: antenna export path
    for getter_name in ("get_antenna_data",):
        getter = getattr(getattr(hfss, "post", hfss), getter_name, None)
        if getter is None:
            getter = getattr(hfss, getter_name, None)
        if getter is None:
            continue
        try:
            ant = getter(
                setup_sweep_name=preferred_setup,
                sphere=context,
                frequencies=[f"{target_freq_ghz}GHz"],
            )
            peak = _peak_from_solution(ant, math_formula="dB")
            if peak is not None:
                return peak
            fail_reasons.append(f"antenna_data({getter_name}): empty")
        except Exception as exc:  # noqa: BLE001
            fail_reasons.append(f"antenna_data({getter_name}): {exc}")

    tail = " | ".join(fail_reasons[-8:]) if fail_reasons else "unknown"
    print(f"[WARN] 提取 {target_freq_ghz} GHz 增益失败。diagnostic={tail}")
    return float("nan")


def _extract_gain_db(hfss: Hfss, target_freq_ghz: float, max_retries: int = 2) -> float:
    for attempt in range(max_retries):
        try:
            gain = _extract_gain_db_once(hfss, target_freq_ghz)
            if np.isfinite(gain) and -60.0 < gain < GAIN_PHYSICAL_MAX:
                return gain
            logging.warning("第%d次提取gain=%.2f无效，等待重试...", attempt + 1, gain)
            time.sleep(2)
        except Exception as exc:  # noqa: BLE001
            logging.warning("第%d次提取增益失败: %s", attempt + 1, exc)
            time.sleep(2)
    return float("nan")


def _save_sim_result(output_dir: str, design_vars: dict[str, Any], result_dict: dict[str, Any]) -> None:
    """将单次仿真结果保存为独立 JSON 文件。"""
    sim_dir = Path(output_dir) / "sim_results"
    sim_dir.mkdir(parents=True, exist_ok=True)

    param_str = json.dumps(design_vars, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = sim_dir / f"sim_{timestamp}_{param_hash}.json"

    payload = {
        "timestamp": timestamp,
        "params": design_vars,
        "result": result_dict,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info("仿真结果已保存: %s", filename)


def _param_signature(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def _is_almost_same_params(
    params_a: dict[str, Any],
    params_b: dict[str, Any],
    tol: float = DEDUP_THRESHOLD,
) -> bool:
    """去重判定：仅当参数几乎完全一致（误差不超过 tol）才认为重复。"""
    if params_a.keys() != params_b.keys():
        return False
    for key in params_a:
        try:
            va = float(params_a[key])
            vb = float(params_b[key])
        except (TypeError, ValueError):
            if params_a[key] != params_b[key]:
                return False
            continue
        if abs(va - vb) > tol:
            return False
    return True


def _evaluate_with_open_hfss(hfss: Hfss, design_vars: DesignVariables, project_path: str) -> tuple[dict[str, Any], Hfss]:
    """在已打开的 HFSS 会话中评估一次设计。"""
    # 只写入优化参数，不覆盖 HFSS 工程内的固定参数
    failed_vars = []
    vars_i = asdict(design_vars)
    design_vars_opt = {k: v for k, v in vars_i.items() if k in OPT_PARAM_NAMES}
    try:
        _apply_design_variables(hfss, design_vars_opt)
    except Exception as exc:  # noqa: BLE001
        logging.warning("写入设计变量首次失败: %s，尝试刷新HFSS会话后重试一次", exc)
        _safe_save(hfss)
        _safe_release(hfss)
        time.sleep(3)
        hfss = _create_hfss_session(Path(project_path), non_graphical=True)
        try:
            _apply_design_variables(hfss, design_vars_opt)
        except Exception as exc2:  # noqa: BLE001
            logging.error("写入设计变量重试仍失败: %s，本轮标记为失败", exc2)
            return {
                "project_name": PROJECT_NAME,
                "design_name": DESIGN_NAME,
                "setup_name": SETUP_NAME,
                "sweep_name": SWEEP_NAME,
                "far_field_sphere": FAR_FIELD_SPHERE,
                "project_path": project_path,
                "design_vars": vars_i,
                "s11_curve": {"freq_ghz": [], "s11_db": []},
                "gain_28ghz_db": float("nan"),
                "gain_38ghz_db": float("nan"),
                "loss": 500.0,
                "dualband_match_ok": False,
                "band_26_32_ok": False,
                "band_37_39_ok": False,
            }, hfss
    _safe_save(hfss)
    time.sleep(1)
    try:
        hfss.delete_solution_data()
    except Exception:
        pass
    try:
        hfss.odesign.DeleteFullVariation("All", False)
    except Exception:
        pass
    if not _check_geometry_valid(hfss):
        raise RuntimeError("几何模型无效，跳过本轮仿真")
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
    if gain_28 > GAIN_PHYSICAL_MAX or gain_28 < -60.0:
        logging.warning("gain_28=%.2f 超出物理范围，置为 nan", gain_28)
        gain_28 = float("nan")
    if gain_38 > GAIN_PHYSICAL_MAX or gain_38 < -60.0:
        logging.warning("gain_38=%.2f 超出物理范围，置为 nan", gain_38)
        gain_38 = float("nan")

    return {
        "project_name": PROJECT_NAME,
        "design_name": DESIGN_NAME,
        "setup_name": SETUP_NAME,
        "sweep_name": SWEEP_NAME,
        "far_field_sphere": FAR_FIELD_SPHERE,
        "project_path": project_path,
        "design_vars": vars_i,
        "s11_curve": {
            "freq_ghz": freqs.tolist(),
            "s11_db": s11_db.tolist(),
        },
        "gain_28ghz_db": gain_28,
        "gain_38ghz_db": gain_38,
        "band_26_32_ok": band1_ok,
        "band_37_39_ok": band2_ok,
        "dualband_match_ok": bool(band1_ok and band2_ok),
    }, hfss


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
    budget: int = 100,
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """调用 SpacePartitioningOptimizer 进行参数优化，并导出结果。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _ensure_numpy_compat()
    _ensure_sklearn_compat()

    try:
        from optimizer import SpacePartitioningOptimizer

        optimizer = SpacePartitioningOptimizer(api_config=API_CONFIG)
        logging.info("优化器初始化成功: SpacePartitioningOptimizer")
    except Exception as e:  # noqa: BLE001
        logging.error("SpacePartitioningOptimizer 初始化失败: %s", e)
        raise RuntimeError("优化器初始化失败，请检查依赖库") from e

    iter_csv = Path(output_dir) / "optimization_log.csv"
    best_json = Path(output_dir) / "best_result.json"
    best_s11_csv = Path(output_dir) / "best_s11_curve.csv"

    best_result: dict[str, Any] | None = None
    best_loss = float("inf")
    default_vars = DesignVariables()

    project_file = Path(project_path).resolve()
    if not project_file.exists():
        raise FileNotFoundError(f"HFSS 工程不存在: {project_file}")

    hfss = _create_hfss_session(project_file, non_graphical=True)

    max_valid_loss: float | None = None
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

            for i in range(1, budget + 1):
                if STOP_REQUESTED:
                    print("[INTERRUPT] 检测到中断请求，提前结束优化循环。")
                    break

                if i > 1 and (i - 1) % SESSION_REFRESH_INTERVAL == 0:
                    logging.info("第 %d 轮，主动刷新HFSS会话...", i)
                    _cleanup_hfss_session(hfss, sleep_sec=1)
                    time.sleep(3)
                    hfss = _create_hfss_session(project_file, non_graphical=True)
                    logging.info("HFSS会话已刷新，新PID已建立")

                cand_raw = optimizer.suggest(1)[0]
                cand = validate_params(cand_raw)
                vars_i = DesignVariables(**{**asdict(default_vars), **cand})

                err_msg = ""
                try:
                    result, hfss = _evaluate_with_open_hfss(hfss, vars_i, str(project_file))
                    if "loss" in result and math.isfinite(float(result["loss"])):
                        loss = float(result["loss"])
                    else:
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

                is_sim_failed = (
                    (not math.isfinite(float(result.get("gain_28ghz_db", float("nan")))))
                    or (not math.isfinite(float(result.get("gain_38ghz_db", float("nan")))))
                    or loss >= 1e6
                )
                if is_sim_failed:
                    logging.warning("仿真失败参数组合: %s", json.dumps(asdict(vars_i), ensure_ascii=False))
                    loss = max_valid_loss if max_valid_loss is not None else 500.0

                optimizer.observe([cand], [loss])

                if not err_msg:
                    _save_sim_result(
                        output_dir,
                        asdict(vars_i),
                        {
                            "gain_28ghz_db": result.get("gain_28ghz_db"),
                            "gain_38ghz_db": result.get("gain_38ghz_db"),
                            "loss": loss,
                            "dualband_match_ok": bool(result.get("dualband_match_ok", False)),
                        },
                    )

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

                if math.isfinite(loss):
                    max_valid_loss = loss if max_valid_loss is None else max(max_valid_loss, loss)

                if not err_msg and loss < best_loss:
                    best_loss = loss
                    best_result = result

                g28 = float(result.get("gain_28ghz_db", float("nan")))
                g38 = float(result.get("gain_38ghz_db", float("nan")))
                dualband_match_ok = bool(result.get("dualband_match_ok", False))
                logging.info(
                    "[%d/%d] loss=%.4f g28=%.4f g38=%.4f dual=%s",
                    i,
                    budget,
                    float(loss),
                    g28,
                    g38,
                    dualband_match_ok,
                )
    finally:
        try:
            _safe_save(hfss)
            _safe_release(hfss)
        except Exception:
            pass

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

    return {
        "best_result_json": str(best_json),
        "best_s11_csv": str(best_s11_csv) if best_result is not None else "",
        "iter_log_csv": str(iter_csv),
        "summary": final,
    }


def main() -> None:
    try:
        import pyaedt

        logging.info("PyAEDT 版本: %s", pyaedt.__version__)
        _kill_stale_aedt()
        opt_result = run_optimization(
            project_path=PROJECT_PATH,
            budget=BUDGET,
            output_dir="outputs",
        )
        print(json.dumps(opt_result, indent=2, ensure_ascii=False))
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 用户主动中断，脚本已安全退出。")


if __name__ == "__main__":
    main()
