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
    这里简单示例，您也可以自定义固定位置。
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
    size 表示方形边长
    """
    (cx, cy), size = obstacle
    half = size / 2.0
    x_min, x_max = cx - half, cx + half
    y_min, y_max = cy - half, cy + half
    x, y = point
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)

def collision_cost(point, obstacles):
    """
    碰撞代价：如果 point 落在任意一个方形障碍物内，则代价很大。
    简单处理：在障碍物内 cost=10.0，否则 cost=0.
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
    E = w_smooth * sum of squared distance between consecutive points
      + w_collision * sum of collision costs
      + w_start * distance^2( x_1, start )
      + w_goal  * distance^2( x_N, goal  )
    """
    E = 0.0
    # 平滑项
    for i in range(len(trajectory) - 1):
        E += w_smooth * np.sum((trajectory[i+1] - trajectory[i])**2)
    # 碰撞项
    for i in range(len(trajectory)):
        E += w_collision * collision_cost(trajectory[i], obstacles)
    # 起点、终点约束
    E += w_start * np.sum((trajectory[0] - start)**2)
    E += w_goal  * np.sum((trajectory[-1] - goal)**2)
    return E

def compute_gradient(trajectory, obstacles, start, goal,
                     w_smooth=1.0, w_collision=10.0, w_start=5.0, w_goal=5.0):
    """
    计算能量函数对轨迹中每个点的梯度 (简化实现)。
    trajectory: shape = (N, 2)
    返回与 trajectory 相同 shape 的梯度数组。
    """
    grad = np.zeros_like(trajectory)

    # 对平滑项求梯度
    for i in range(len(trajectory)):
        if i > 0:
            grad[i] += 2 * w_smooth * (trajectory[i] - trajectory[i-1])
            grad[i-1] -= 2 * w_smooth * (trajectory[i] - trajectory[i-1])
        if i < len(trajectory) - 1:
            grad[i] += 2 * w_smooth * (trajectory[i] - trajectory[i+1])
            grad[i+1] -= 2 * w_smooth * (trajectory[i] - trajectory[i+1])

    # 对碰撞项求梯度（这里非常简化：如果在障碍物内，就给当前点一个随机的正向梯度）
    # 实际中可以用障碍物与点的最近距离来计算更平滑的梯度。
    for i in range(len(trajectory)):
        c = collision_cost(trajectory[i], obstacles)
        if c > 0:  # 在障碍物中
            # 给出一个将点推离障碍物中心的梯度
            # 找到最近的障碍物中心
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
                    direction = np.random.randn(2)  # 万一重合就随机一个方向
                direction = direction / (np.linalg.norm(direction) + 1e-9)
                grad[i] += w_collision * 20.0 * direction  # 乘以一个系数把它推开

    # 对起点、终点约束求梯度
    grad[0] += 2 * w_start * (trajectory[0] - start)
    grad[-1] += 2 * w_goal  * (trajectory[-1] - goal)

    return grad

# -----------------------
# 3. 轨迹去噪过程
# -----------------------

def denoise_trajectory(trajectory, obstacles, start, goal, 
                       num_iterations=100, lr=0.01):
    """
    对轨迹进行若干步梯度下降，返回在各个指定时间步的轨迹列表
    """
    # 准备记录不同迭代步的轨迹，用于可视化
    # 假设我们想在 [100, 80, 60, 40, 20, 5, 0] 这些步数上可视化
    save_steps = [100, 80, 60, 40, 20, 5, 0]
    results = {}

    for s in range(num_iterations, -1, -1):  # 从 num_iterations 到 0
        # 计算梯度
        grad = compute_gradient(trajectory, obstacles, start, goal)
        # 梯度下降更新
        trajectory -= lr * grad

        # 如果 s 在我们的保存列表里，就保存当前轨迹
        if s in save_steps:
            results[s] = trajectory.copy()

    return results

# -----------------------
# 4. 主函数：生成环境 + 可视化
# -----------------------
def main():
    np.random.seed(0)  # 固定随机种子，便于复现

    # 4.1 生成随机障碍物
    obstacles = generate_square_obstacles(num_obstacles=5)

    # 4.2 定义起点和终点
    start = np.array([0.0, 0.0])
    goal  = np.array([10.0, 10.0])

    # 4.3 生成随机初始轨迹 (例如 10 个中间离散点)
    N = 10
    trajectory = np.random.randn(N, 2) * 2.0 + (5.0, 5.0)  # 随机中心在(5,5)附近

    # 4.4 进行去噪（梯度下降）
    results = denoise_trajectory(trajectory, obstacles, start, goal, 
                                 num_iterations=100, lr=0.01)

    # 4.5 画图展示
    # 我们要展示 S=100,80,60,40,20,5,0 共7个子图
    steps_to_plot = [100, 80, 60, 40, 20, 5, 0]
    fig, axes = plt.subplots(1, len(steps_to_plot), figsize=(18, 3), dpi=100)

    for idx, s in enumerate(steps_to_plot):
        ax = axes[idx]

        ax.set_title(f"S={s}")
        # 设置标题在下方
        ax.title.set_position([0.5, 10])

        # 绘制障碍物(蓝色方块)
        for obs in obstacles:
            (cx, cy), size = obs
            half = size / 2.0
            square = plt.Rectangle((cx-half, cy-half), size, size, 
                                   color='blue', alpha=0.5)
            ax.add_patch(square)

        # 绘制轨迹
        traj_s = results[s]
        ax.plot(traj_s[:,0], traj_s[:,1], '-o', color='red', markersize=3)

        # 绘制起点和终点
        ax.plot(start[0], start[1], 'gs', label='start')
        ax.plot(goal[0], goal[1], 'g*', label='goal')
        # 画 `is success` 指示灯 (S=0 时绿色，其余为红色)
        success_color = 'green' if s == 0 else 'red'

        ax.text(0.1, 9.5, "is success:", fontsize=10, fontweight='bold')
        ax.add_patch(patches.Rectangle((2, 9.3), 0.5, 0.5, color=success_color))

        # 画 `len: 48`
        ax.text(2.7, 9.5, "len: 48", fontsize=10, fontweight='bold')

        # 坐标范围
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.set_aspect('equal')
        ax.grid(True)

    plt.tight_layout()
    # plt.grid(True)
    plt.savefig("paintmultifig.png")
    plt.show()

if __name__ == "__main__":
    main()
