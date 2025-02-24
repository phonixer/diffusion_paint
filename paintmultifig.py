import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------
# 1. 定义障碍物及辅助函数
# -----------------------

def generate_square_obstacles(num_obstacles=5, 
                              x_range=(1,9), 
                              y_range=(1,9), 
                              size=1.0):
    """
    随机生成若干方形障碍物。
    返回：[(x_center, y_center), size, ...]
    """
    obstacles = []
    for _ in range(num_obstacles):
        cx = np.random.uniform(x_range[0], x_range[1])
        cy = np.random.uniform(y_range[0], y_range[1])
        obstacles.append(((cx, cy), size))
    return obstacles

def in_square_obstacle(point, obstacle):
    """
    判断 point 是否在给定 obstacle (方形) 内。
    obstacle = ((cx, cy), size)
    """
    (cx, cy), size = obstacle
    half = size / 2.0
    x_min, x_max = cx - half, cx + half
    y_min, y_max = cy - half, cy + half
    x, y = point
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)

def collision_cost(point, obstacles):
    """
    如果 point 落在任一障碍物内，返回较大的代价；否则返回0.
    """
    for obs in obstacles:
        if in_square_obstacle(point, obs):
            return 10.0
    return 0.0

# -----------------------
# 2. 定义能量函数及梯度
# -----------------------

def compute_energy(trajectory, obstacles, start, goal,
                   w_smooth=1.0, w_collision=10.0, w_start=5.0, w_goal=5.0):
    """
    简易能量函数：
      E = w_smooth * 连续点间的平方距离和
        + w_collision * 碰撞代价和
        + w_start * 起点约束
        + w_goal  * 终点约束
    """
    E = 0.0
    for i in range(len(trajectory) - 1):
        E += w_smooth * np.sum((trajectory[i+1] - trajectory[i])**2)
    for i in range(len(trajectory)):
        E += w_collision * collision_cost(trajectory[i], obstacles)
    E += w_start * np.sum((trajectory[0] - start)**2)
    E += w_goal  * np.sum((trajectory[-1] - goal)**2)
    return E

def compute_gradient(trajectory, obstacles, start, goal,
                     w_smooth=1.0, w_collision=10.0, w_start=5.0, w_goal=5.0):
    """
    计算能量函数对轨迹中各点的梯度（简化实现）。
    """
    grad = np.zeros_like(trajectory)
    for i in range(len(trajectory)):
        if i > 0:
            grad[i] += 2 * w_smooth * (trajectory[i] - trajectory[i-1])
            grad[i-1] -= 2 * w_smooth * (trajectory[i] - trajectory[i-1])
        if i < len(trajectory) - 1:
            grad[i] += 2 * w_smooth * (trajectory[i] - trajectory[i+1])
            grad[i+1] -= 2 * w_smooth * (trajectory[i] - trajectory[i+1])
    for i in range(len(trajectory)):
        c = collision_cost(trajectory[i], obstacles)
        if c > 0:
            nearest_obs_center = None
            nearest_dist = float('inf')
            for obs in obstacles:
                (cx, cy), size = obs
                dist = np.hypot(trajectory[i,0] - cx, trajectory[i,1] - cy)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_obs_center = (cx, cy)
            if nearest_obs_center is not None:
                direction = trajectory[i] - np.array(nearest_obs_center)
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.random.randn(2)
                direction = direction / (np.linalg.norm(direction) + 1e-9)
                grad[i] += w_collision * 20.0 * direction
    grad[0] += 2 * w_start * (trajectory[0] - start)
    grad[-1] += 2 * w_goal  * (trajectory[-1] - goal)
    return grad

# -----------------------
# 3. 轨迹去噪过程
# -----------------------

def denoise_trajectory(trajectory, obstacles, start, goal, 
                       num_iterations=100, lr=0.01):
    """
    对轨迹进行梯度下降去噪，并保存特定迭代步的结果。
    """
    save_steps = [100, 80, 60, 40, 20, 5, 0]
    results = {}
    for s in range(num_iterations, -1, -1):
        grad = compute_gradient(trajectory, obstacles, start, goal)
        trajectory -= lr * grad
        if s in save_steps:
            results[s] = trajectory.copy()
    return results

# -----------------------
# 4. 主函数：生成环境 + 可视化
# -----------------------

def main():
    np.random.seed(0)
    obstacles = generate_square_obstacles(num_obstacles=5)
    start = np.array([0.0, 0.0])
    goal  = np.array([10.0, 10.0])
    N = 10
    trajectory = np.random.randn(N, 2) * 2.0 + (5.0, 5.0)
    results = denoise_trajectory(trajectory, obstacles, start, goal, num_iterations=100, lr=0.01)

    steps_to_plot = [100, 80, 60, 40, 20, 5, 0]
    fig, axes = plt.subplots(1, len(steps_to_plot), figsize=(18, 10), dpi=100)
    fig.suptitle("Trajectory Denoising Process", fontsize=16, fontweight='bold', y=1.05)

    for idx, s in enumerate(steps_to_plot):
        ax = axes[idx]
        # 绘制障碍物（蓝色方块）
        for obs in obstacles:
            (cx, cy), size = obs
            half = size / 2.0
            square = plt.Rectangle((cx-half, cy-half), size, size, facecolor='#4C72B0', alpha=0.5, edgecolor='black')
            ax.add_patch(square)
        # 绘制轨迹（红色连线和圆点）
        traj_s = results[s]
        ax.plot(traj_s[:,0], traj_s[:,1], '-o', color='red', markersize=3)
        # 绘制起点和终点
        ax.plot(start[0], start[1], 'gs', markersize=6)
        ax.plot(goal[0], goal[1], 'g*', markersize=10)
        # 坐标与网格
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.set_aspect('equal')
        ax.grid(True)
        
        # -----------------------
        # 在图像下方添加小标题和指示信息（使用 ax.transAxes 坐标）
        # 图上方高度
        up_y = 1.05

        # -----------------------
        # 小标题 S=xxx 放在图像正下方
        ax.text(0.5, -0.15, f"S={s}", fontsize=10, fontweight='bold',
                transform=ax.transAxes, ha='center', va='top', clip_on=False)
        # “is success:” 及色块和 "len: 48" 放在小标题更下方
        ax.text(0.05, up_y, "is success:", fontsize=10, fontweight='bold',
                transform=ax.transAxes, ha='left', va='center', clip_on=False)
        # 色块：S=0 时为绿色，其余为红色
        success_color = 'green' if s == 0 else 'red'
        # 这里利用 patches.Rectangle，并指定 transform 为 ax.transAxes
        rect = patches.Rectangle((-0.05, up_y), 0.05, 0.05, transform=ax.transAxes,
                         facecolor=success_color, edgecolor='black', clip_on=False)

        


        ax.add_patch(rect)
        # "len: 48" 文本放在色块右侧
        ax.text(0.7, up_y, "len: 48", fontsize=10, fontweight='bold',
                transform=ax.transAxes, ha='left', va='center')
        




    
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("paintmultifig.png")
    plt.show()

if __name__ == "__main__":
    main()
