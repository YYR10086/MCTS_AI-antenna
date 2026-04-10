"""
tesths.py
=========
用于测试本次 TuRBO 异质方差改造（HEBO-style scoring）的关键逻辑。

设计目标：
1) 不依赖 HFSS，不调用真实电磁仿真；
2) 尽量避免依赖 bayesmark，聚焦 turbo1.py 内新增逻辑；
3) 既测试“功能”，也做一层“代码审计”（配置字段是否接通）。
"""

import ast
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


def _make_turbo_for_unit_test(dim=2):
    """构造一个最小可用 Turbo1 实例（不会调用真实目标函数）。"""
    from turbo1 import Turbo1

    lb = np.zeros(dim)
    ub = np.ones(dim)
    return Turbo1(
        f=lambda x: float(np.sum(x**2)),
        lb=lb,
        ub=ub,
        n_init=4,
        max_evals=30,
        batch_size=1,
        verbose=False,
        use_hetero_lcb=1,
        budget=30,
    )


class TestHeteroTurbo(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy 未安装，跳过数值测试")
    def test_estimate_local_noise_higher_in_noisy_region(self):
        """
        构造两团数据：左团低方差，右团高方差。
        期望 _estimate_local_noise 在右团 query 的估计值更高。
        """
        turbo = _make_turbo_for_unit_test(dim=2)
        rng = np.random.default_rng(1234)

        # 左团: 方差小
        x_left = np.clip(rng.normal(loc=[0.2, 0.2], scale=0.03, size=(40, 2)), 0.0, 1.0)
        y_left = 0.2 + rng.normal(0.0, 0.01, size=40)

        # 右团: 方差大
        x_right = np.clip(rng.normal(loc=[0.8, 0.8], scale=0.03, size=(40, 2)), 0.0, 1.0)
        y_right = 0.2 + rng.normal(0.0, 0.15, size=40)

        X_train = np.vstack([x_left, x_right])
        y_train = np.concatenate([y_left, y_right])

        # 注意：_estimate_local_noise 的归一化是“对本次 query 集合内部”做的，
        # 因此必须一次性同时传入两侧 query 才能做相对比较。
        q_left = np.clip(rng.normal(loc=[0.2, 0.2], scale=0.01, size=(8, 2)), 0.0, 1.0)
        q_right = np.clip(rng.normal(loc=[0.8, 0.8], scale=0.01, size=(8, 2)), 0.0, 1.0)
        q_all = np.vstack([q_left, q_right])
        noise_all = turbo._estimate_local_noise(X_train, y_train, q_all)
        n_left = float(np.mean(noise_all[: len(q_left)]))
        n_right = float(np.mean(noise_all[len(q_left) :]))

        self.assertGreater(
            n_right,
            n_left,
            msg=f"异质噪声估计异常：右团(高噪)={n_right:.4f} 不大于 左团(低噪)={n_left:.4f}",
        )

    @unittest.skipIf(np is None, "numpy 未安装，跳过数值测试")
    def test_select_candidates_penalizes_local_noise(self):
        """
        当候选 y 均值与方差接近时，异质噪声惩罚应优先选择低噪声点。
        """
        turbo = _make_turbo_for_unit_test(dim=2)
        turbo.batch_size = 1
        turbo.used_budget = 10
        turbo.f_var = np.array([0.04, 0.04, 0.04])  # 方差一致
        turbo.local_noise_cand = np.array([0.1, 2.0, 1.8])  # 第0个噪声最低

        X_cand = np.array([[0.1, 0.1], [0.7, 0.7], [0.9, 0.9]])
        y_cand = np.array([[0.5], [0.5], [0.5]])  # 均值一致

        chosen = turbo._select_candidates(X_cand, y_cand)
        np.testing.assert_allclose(
            chosen[0],
            X_cand[0],
            atol=1e-12,
            err_msg="异质噪声惩罚未生效：未优先选中低噪声候选点。",
        )

    @unittest.skipIf(np is None, "numpy 未安装，跳过数值测试")
    def test_optimizer_config_contains_hetero_keys(self):
        """
        轻量代码审计：直接解析 optimizer.py，确认新参数写入默认配置。
        （避免 import optimizer 带来的 bayesmark 依赖问题）
        """
        with open("optimizer.py", "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        keys_found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in {
                    "use_hetero_lcb",
                    "hetero_beta0",
                    "hetero_beta1",
                    "hetero_noise_penalty",
                    "hetero_k_neighbors",
                }:
                    keys_found.add(node.value)

        self.assertEqual(
            keys_found,
            {"use_hetero_lcb", "hetero_beta0", "hetero_beta1", "hetero_noise_penalty", "hetero_k_neighbors"},
            msg=f"配置项缺失，当前检测到: {sorted(keys_found)}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
