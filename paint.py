import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 假设扩散步骤数据
steps = [0, 5, 10, 15, 20]
steps_interval = 5  # 每个步骤包含的路径数量

# 模拟地面真值轨迹
ground_truth = np.column_stack((np.linspace(6.5, 9, 100), np.linspace(320, 460, 100)))

# 初始化数据列表
data = []

# 为每个扩散步骤生成带噪声的数据
for step in steps:
    traj = []  # 保存当前步骤的路径数据
    for j in range(steps_interval):
        noisy_traj = ground_truth + np.random.normal(0, 0.1 * (max(steps) -step - j + 5), ground_truth.shape)  # 加入噪声
        traj.append(noisy_traj)
    data.append(np.vstack(traj))  # 将多条路径拼接为一个整体数据

# 创建绘图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 绘制每个扩散步骤的密度图
for i, step in enumerate(steps):
    ax = axes[i]
    sns.kdeplot(
        x=data[i][:, 0],  # 所有路径的 x 坐标
        y=data[i][:, 1],  # 所有路径的 y 坐标
        fill=True,
        cmap='Blues',
        ax=ax
    )
    ax.set_title(f"(a) Diffusion Step {step}", fontsize=12)
    ax.set_xlim(6, 10)
    ax.set_ylim(300, 470)
    ax.set_xlabel("Longitudinal Distance (m)")
    ax.set_ylabel("Latitudinal Distance (m)")

# 绘制地面真值轨迹
axes[-1].plot(ground_truth[:, 0], ground_truth[:, 1], color="blue", linewidth=2, label="Ground Truth")
axes[-1].set_title("(f) Ground Truth", fontsize=12)
axes[-1].set_xlim(6, 10)
axes[-1].set_ylim(300, 470)
axes[-1].set_xlabel("Longitudinal Distance (m)")
axes[-1].set_ylabel("Latitudinal Distance (m)")
axes[-1].legend()

# 调整布局并保存图片
plt.tight_layout()
plt.savefig("rou_diffusion.png")
plt.show()
