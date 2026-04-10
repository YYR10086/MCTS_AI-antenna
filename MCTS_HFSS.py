import os
import sys
import csv
import numpy as np
from pyaedt import Hfss

sys.path.append(os.getcwd())

try:
    import submissions.space_decay.optimizer as space_decay_optimizer
except ImportError:
    import optimizer as space_decay_optimizer

SpacePartitioningOptimizer = space_decay_optimizer.SpacePartitioningOptimizer

if hasattr(space_decay_optimizer, "DEBUG"):
    space_decay_optimizer.DEBUG = False

original_read_config = SpacePartitioningOptimizer._read_config

def patched_read_config(self):
    conf = original_read_config(self)
    conf["reset_no_improvement"] = 1000000
    conf["n_init_points"] = 6  # 3D 参数空间里更快进入模型阶段
    conf["turbo_training_steps"] = 80
    conf["turbo"]["budget"] = 20
    conf["turbo"]["use_hetero_lcb"] = 1
    conf["turbo"]["hetero_beta0"] = 2.2
    conf["turbo"]["hetero_beta1"] = 0.8
    conf["turbo"]["hetero_noise_penalty"] = 0.35
    conf["turbo"]["hetero_k_neighbors"] = 6
    conf["debug"] = False
    conf["DEBUG"] = False
    return conf

SpacePartitioningOptimizer._read_config = patched_read_config
print("✓ 已禁用优化器重置逻辑\n")

# ================= 配置区 =================
PROJECT_PATH = os.path.abspath(r"ceramic_monoblock_MMDS_Band.aedt")
SETUP_NAME = "Setup_5GNR_Band_N41"
SWEEP_NAME = f"{SETUP_NAME} : LastAdaptive"   # 如有真实 Sweep，建议替换
TARGET_FREQ_GHZ = 2.6                         # 改成你的目标频点
CSV_LOG = "hfss_optimization_log.csv"

if not os.path.exists(PROJECT_PATH):
    print(f"错误：项目文件不存在: {PROJECT_PATH}")
    sys.exit(1)

# ================= HFSS 环境 =================
print("正在连接 HFSS...")
hfss = Hfss(project=PROJECT_PATH, version="2025.1")
print("HFSS 连接成功！\n")

def extract_target_s11(solution_data, target_freq_ghz):
    """
    从 HFSS 返回的整条 S11 曲线中提取目标频点的值
    """
    freqs = np.array(solution_data.primary_sweep_values, dtype=float)
    vals = np.array(solution_data.data_real(), dtype=float)

    if len(freqs) == 0 or len(vals) == 0:
        raise ValueError("未获取到有效的 S11 曲线数据。")

    idx = np.argmin(np.abs(freqs - target_freq_ghz))
    return float(vals[idx]), float(freqs[idx])

def simulate(params_dict):
    try:
        # 更新参数
        hfss["res1_capY"] = f"{params_dict['res1_capY']:.4f}mm"
        hfss["gap1to2"] = f"{params_dict['gap1to2']:.4f}mm"
        hfss["input_gap"] = f"{params_dict['input_gap']:.4f}mm"

        # 运行仿真
        hfss.analyze_setup(SETUP_NAME)

        # 读取结果
        sol = hfss.post.get_solution_data(
            expressions="dB(S(1,1))",
            setup_sweep_name=SWEEP_NAME
        )

        if sol is None:
            raise ValueError("get_solution_data 返回 None。")

        s11, actual_freq = extract_target_s11(sol, TARGET_FREQ_GHZ)
        print(f"  目标频点附近 S11: {s11:.3f} dB @ {actual_freq:.6f} GHz")
        return float(s11), True

    except Exception as e:
        print(f"  仿真失败: {e}")
        # 最小化问题里，失败给一个大惩罚
        return 100.0, False

def log_result(iter_id, params, loss, success):
    file_exists = os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iter", "res1_capY", "gap1to2", "input_gap", "s11_db", "success"])
        writer.writerow([
            iter_id,
            params["res1_capY"],
            params["gap1to2"],
            params["input_gap"],
            loss,
            int(success)
        ])

# ================= 运行优化 =================
if __name__ == "__main__":
    api_config = {
        "res1_capY": {"type": "real", "space": "linear", "range": (1.2, 1.8)},
        "gap1to2": {"type": "real", "space": "linear", "range": (0.005, 0.025)},
        "input_gap": {"type": "real", "space": "linear", "range": (0.01, 0.05)}
    }

    print("========== 开始 MCTS-TuRBO + HFSS 优化 ==========\n")

    optimizer = SpacePartitioningOptimizer(api_config=api_config)

    BUDGET = 20
    best_s11 = float("inf")
    best_params = None

    try:
        for i in range(BUDGET):
            suggestions = optimizer.suggest(1)
            s = suggestions[0]

            print(f"[第 {i + 1}/{BUDGET} 次仿真] 参数: {s}")
            loss, success = simulate(s)

            log_result(i + 1, s, loss, success)

            if success and loss < best_s11:
                best_s11 = loss
                best_params = dict(s)

            optimizer.observe(suggestions, [loss])

            if best_params is not None:
                print(f"  当前最优 S11: {best_s11:.3f} dB")
                print(f"  当前最优参数: {best_params}\n")
            else:
                print("  当前还没有成功样本。\n")

    finally:
        print("正在释放 HFSS...")
        try:
            hfss.release_desktop()
        except Exception:
            pass

    print("========== 优化完成 ==========")
    if best_params is not None:
        print(f"最优参数: {best_params}")
        print(f"最优 S11: {best_s11:.3f} dB")
    else:
        print("没有得到有效的成功仿真结果。")
