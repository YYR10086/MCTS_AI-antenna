import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
from pyaedt import Hfss

sys.path.append(os.getcwd())

# 兼容 bayesmark 在 NumPy 2.x 下对 np.float_ 的旧引用
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

try:
    import submissions.space_decay.optimizer as space_decay_optimizer
except ImportError:
    import optimizer as space_decay_optimizer

SpacePartitioningOptimizer = space_decay_optimizer.SpacePartitioningOptimizer
if hasattr(space_decay_optimizer, "DEBUG"):
    space_decay_optimizer.DEBUG = False


# ========= HFSS 工程信息（按你的要求） =========
PROJECT_NAME = "A1"
DESIGN_NAME = "HFSSDesign1"
SETUP_NAME = "Setup1"
SWEEP_NAME = "Sweep"
FAR_FIELD_SPHERE = "Infinite Sphere1"
PROJECT_PATH = os.path.abspath("A1.aedt")  # 完整路径占位，可改成绝对路径

CSV_LOG = "hfss_dualband_optimization_log.csv"
FINAL_JSON = "hfss_dualband_final_result.json"

# 频段要求
BAND1 = (26.0, 32.0)  # GHz
BAND2 = (37.0, 39.0)  # GHz
TARGET_GAINS = (28.0, 38.0)  # GHz
S11_THRESHOLD_DB = -10.0


@dataclass
class DesignVariables:
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


def patch_optimizer_config():
    """使用你之前做好的 hetero-TuRBO 算法配置。"""
    original_read_config = SpacePartitioningOptimizer._read_config

    def patched_read_config(self):
        conf = original_read_config(self)
        conf["reset_no_improvement"] = 1000000
        conf["turbo_training_steps"] = 80
        conf["turbo"]["use_hetero_lcb"] = 1
        conf["turbo"]["hetero_beta0"] = 2.2
        conf["turbo"]["hetero_beta1"] = 0.8
        conf["turbo"]["hetero_noise_penalty"] = 0.35
        conf["turbo"]["hetero_k_neighbors"] = 6
        return conf

    SpacePartitioningOptimizer._read_config = patched_read_config


def apply_design_variables(hfss: Hfss, dv: DesignVariables):
    """将设计变量写入 HFSS（mm）。"""
    for key, val in asdict(dv).items():
        hfss[key] = f"{val:.6f}mm"


def get_s11_curve(hfss: Hfss) -> Tuple[np.ndarray, np.ndarray]:
    setup_sweep = f"{SETUP_NAME} : {SWEEP_NAME}"
    sol = hfss.post.get_solution_data(expressions="dB(S(1,1))", setup_sweep_name=setup_sweep)
    if sol is None:
        raise RuntimeError("未能提取 S11 曲线（get_solution_data 返回 None）。")
    freqs = np.array(sol.primary_sweep_values, dtype=float)
    s11_db = np.array(sol.data_real(), dtype=float)
    if freqs.size == 0 or s11_db.size == 0:
        raise RuntimeError("S11 曲线为空。")
    return freqs, s11_db


def band_match_ok(freqs: np.ndarray, s11_db: np.ndarray, band: Tuple[float, float], threshold_db=-10.0):
    fmin, fmax = band
    m = (freqs >= fmin) & (freqs <= fmax)
    if np.sum(m) == 0:
        return False
    return bool(np.all(s11_db[m] < threshold_db))


def extract_gain_at_freq(hfss: Hfss, target_freq_ghz: float) -> float:
    """
    提取指定频点增益（基于 Infinite Sphere1）。
    若场景表达式差异导致失败，返回 nan。
    """
    setup_sweep = f"{SETUP_NAME} : {SWEEP_NAME}"
    try:
        sol = hfss.post.get_solution_data(
            expressions="dB(GainTotal)",
            setup_sweep_name=setup_sweep,
            context=FAR_FIELD_SPHERE,
        )
        if sol is None:
            return float("nan")
        freqs = np.array(sol.primary_sweep_values, dtype=float)
        vals = np.array(sol.data_real(), dtype=float)
        if freqs.size == 0 or vals.size == 0:
            return float("nan")
        idx = int(np.argmin(np.abs(freqs - target_freq_ghz)))
        return float(vals[idx])
    except Exception:
        return float("nan")


def evaluate_design(hfss: Hfss, dv: DesignVariables) -> Dict:
    """
    运行仿真并返回完整结果：
      - S11 曲线
      - 28/38GHz 增益
      - 双频段是否满足 |S11|<-10dB
    """
    apply_design_variables(hfss, dv)
    hfss.analyze_setup(SETUP_NAME)

    freqs, s11_db = get_s11_curve(hfss)
    b1_ok = band_match_ok(freqs, s11_db, BAND1, S11_THRESHOLD_DB)
    b2_ok = band_match_ok(freqs, s11_db, BAND2, S11_THRESHOLD_DB)
    dualband_ok = bool(b1_ok and b2_ok)

    gain_28 = extract_gain_at_freq(hfss, TARGET_GAINS[0])
    gain_38 = extract_gain_at_freq(hfss, TARGET_GAINS[1])

    return {
        "design_vars": asdict(dv),
        "s11_curve": {
            "freq_ghz": freqs.tolist(),
            "s11_db": s11_db.tolist(),
        },
        "gain_28ghz_db": gain_28,
        "gain_38ghz_db": gain_38,
        "band_26_32_ok": b1_ok,
        "band_37_39_ok": b2_ok,
        "dualband_match_ok": dualband_ok,
    }


def objective_from_result(result: Dict) -> float:
    """
    优化目标（越小越好）：
      1) 频段不满足时加大惩罚
      2) 次目标：提高 28/38GHz 增益（通过负号转成最小化）
      3) 次目标：全局 S11 最小值越低越好
    """
    s11 = np.array(result["s11_curve"]["s11_db"], dtype=float)
    freqs = np.array(result["s11_curve"]["freq_ghz"], dtype=float)
    g28 = float(result["gain_28ghz_db"])
    g38 = float(result["gain_38ghz_db"])

    penalty = 0.0
    if not result["band_26_32_ok"]:
        m = (freqs >= BAND1[0]) & (freqs <= BAND1[1])
        if np.any(m):
            penalty += float(np.mean(np.maximum(s11[m] - S11_THRESHOLD_DB, 0.0))) * 20.0
        else:
            penalty += 500.0
    if not result["band_37_39_ok"]:
        m = (freqs >= BAND2[0]) & (freqs <= BAND2[1])
        if np.any(m):
            penalty += float(np.mean(np.maximum(s11[m] - S11_THRESHOLD_DB, 0.0))) * 20.0
        else:
            penalty += 500.0

    gain_term = 0.0
    if np.isfinite(g28):
        gain_term -= 0.5 * g28
    if np.isfinite(g38):
        gain_term -= 0.5 * g38

    s11_term = float(np.min(s11))
    return penalty + gain_term + s11_term


def export_iteration_csv(iter_id: int, result: Dict, loss: float):
    file_exists = os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(
                [
                    "iter",
                    "loss",
                    "gain_28ghz_db",
                    "gain_38ghz_db",
                    "band_26_32_ok",
                    "band_37_39_ok",
                    "dualband_match_ok",
                    "design_vars_json",
                ]
            )
        w.writerow(
            [
                iter_id,
                loss,
                result["gain_28ghz_db"],
                result["gain_38ghz_db"],
                int(result["band_26_32_ok"]),
                int(result["band_37_39_ok"]),
                int(result["dualband_match_ok"]),
                json.dumps(result["design_vars"], ensure_ascii=False),
            ]
        )


def main():
    patch_optimizer_config()

    if not os.path.exists(PROJECT_PATH):
        raise FileNotFoundError(f"HFSS 工程不存在：{PROJECT_PATH}")

    print("连接 HFSS...")
    hfss = None
    best_result = None
    best_loss = float("inf")

    # 使用你之前的算法，仅优化部分变量，其余保持给定默认值
    api_config = {
        "Rc": {"type": "real", "space": "linear", "range": (5.5, 7.5)},
        "S": {"type": "real", "space": "linear", "range": (0.6, 1.8)},
        "dp": {"type": "real", "space": "linear", "range": (0.2, 0.7)},
        "x1": {"type": "real", "space": "linear", "range": (1.0, 2.5)},
        "y1": {"type": "real", "space": "linear", "range": (1.0, 2.2)},
    }

    optimizer = SpacePartitioningOptimizer(api_config=api_config)
    default_vars = DesignVariables()
    budget = 20

    try:
        hfss = Hfss(
            project=PROJECT_PATH,
            design=DESIGN_NAME,
            version="2025.1",
            new_desktop_session=False,
        )

        for i in range(1, budget + 1):
            cand = optimizer.suggest(1)[0]
            dv = DesignVariables(**{**asdict(default_vars), **cand})
            try:
                result = evaluate_design(hfss, dv)
                loss = objective_from_result(result)
                success = True
            except Exception as e:
                result = {
                    "design_vars": asdict(dv),
                    "s11_curve": {"freq_ghz": [], "s11_db": []},
                    "gain_28ghz_db": float("nan"),
                    "gain_38ghz_db": float("nan"),
                    "band_26_32_ok": False,
                    "band_37_39_ok": False,
                    "dualband_match_ok": False,
                    "error": str(e),
                }
                loss = 1e6
                success = False

            export_iteration_csv(i, result, loss)
            optimizer.observe([cand], [loss])

            if success and loss < best_loss:
                best_loss = loss
                best_result = result

            print(
                f"[{i}/{budget}] loss={loss:.4f}, dualband_ok={result.get('dualband_match_ok', False)}, "
                f"g28={result.get('gain_28ghz_db')}, g38={result.get('gain_38ghz_db')}"
            )

    except Exception as e:
        print(f"运行异常：{e}")
        raise
    finally:
        if hfss is not None:
            try:
                hfss.release_desktop()
            except Exception:
                pass

    if best_result is None:
        best_result = {
            "message": "没有成功仿真结果",
            "project_path": PROJECT_PATH,
            "project_name": PROJECT_NAME,
            "design_name": DESIGN_NAME,
            "setup_name": SETUP_NAME,
            "sweep_name": SWEEP_NAME,
            "far_field_sphere": FAR_FIELD_SPHERE,
        }
    else:
        best_result["project_path"] = PROJECT_PATH
        best_result["project_name"] = PROJECT_NAME
        best_result["design_name"] = DESIGN_NAME
        best_result["setup_name"] = SETUP_NAME
        best_result["sweep_name"] = SWEEP_NAME
        best_result["far_field_sphere"] = FAR_FIELD_SPHERE
        best_result["best_loss"] = best_loss

    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(best_result, f, ensure_ascii=False, indent=2)
    print(f"结果已导出：{FINAL_JSON}")


if __name__ == "__main__":
    main()
