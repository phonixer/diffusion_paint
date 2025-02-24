import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 生成示例轨迹数据（假设时间间隔均匀）
np.random.seed(42)
t = np.linspace(0, 10, 500)  # 500个时间步
x = np.cumsum(np.random.randn(len(t)))  # 随机轨迹
y = np.cumsum(np.random.randn(len(t)))

# 计算速度 (v_x, v_y)
v_x = np.gradient(x, t)
v_y = np.gradient(y, t)
v_longitudinal = np.sqrt(v_x**2 + v_y**2)  # 计算总速度
v_lateral = np.arctan2(v_y, v_x)  # 计算横向速度（角度）

# 计算加速度 (a_x, a_y)
a_x = np.gradient(v_x, t)
a_y = np.gradient(v_y, t)
jerk_longitudinal = np.gradient(a_x, t)  # 纵向jerk
jerk_lateral = np.gradient(a_y, t)  # 横向jerk

# --------------- 画图 ------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8))


# 速度分布（小提琴图）
sns.violinplot(data=[v_longitudinal], ax=axes[0, 0], palette="Blues")
axes[0, 0].set_xticks([0])  # 先设置刻度位置
axes[0, 0].set_xticklabels(["Longitudinal Speed"])
axes[0, 0].set_ylabel("Speed (m/s)")
axes[0, 0].set_title("(a) Longitudinal Speed Distribution")

sns.violinplot(data=[v_lateral], ax=axes[0, 1], palette="Blues")
axes[0, 1].set_xticks([0])  # 先设置刻度位置
axes[0, 1].set_xticklabels(["Lateral Speed"])
axes[0, 1].set_ylabel("Speed (m/s)")
axes[0, 1].set_title("(b) Lateral Speed Distribution")

# jerk 分布（直方图+密度曲线）
sns.histplot(jerk_longitudinal, bins=30, kde=True, ax=axes[1, 0], color="blue")
axes[1, 0].set_title("(c) Longitudinal Jerk Distribution")
axes[1, 0].set_ylabel("Probability")

sns.histplot(jerk_lateral, bins=30, kde=True, ax=axes[1, 1], color="green")
axes[1, 1].set_title("(d) Lateral Jerk Distribution")
axes[1, 1].set_ylabel("Probability")

plt.savefig("violin_plot.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()
