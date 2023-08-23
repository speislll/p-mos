import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

import numpy as np


def generate_fracture(n, m, theta, beta):
    """递归生成分形裂缝"""

    # 生成主裂缝
    x = [0, 0, 1, 1]
    y = [0, 1, 1, 0]

    fractures = [(x, y)]

    for i in range(n):
        new_fractures = []
        for fracture in fractures:
            x, y = fracture
            new_x = []
            new_y = []
            for j in range(len(x) - 1):
                x1 = x[j]
                y1 = y[j]
                x2 = x[j + 1]
                y2 = y[j + 1]

                # 计算分支裂缝端点坐标
                for k in range(m):
                    theta_k = theta[k]
                    beta_k = beta[k]
                    new_x.append(x1 + beta_k * (x2 - x1) * np.cos(theta_k))
                    new_y.append(y1 + beta_k * (y2 - y1) * np.sin(theta_k))

                new_x.append(x2)
                new_y.append(y2)

            # 添加分支裂缝
            new_fractures.append((new_x, new_y))

        fractures = new_fractures

    return fractures


import numpy as np
from generate_fracture import generate_fracture


def create_fractures(n, m, theta, beta):
    """生成裂缝网络"""

    fractures = []

    # 生成左右两个簇裂缝
    for i in [0, 1]:
        # 设定偏转角和缩放因子
        if i == 0:
            theta = [np.pi / 4, np.pi / 6]
            beta = [0.8, 1.2]
        else:
            theta = [-np.pi / 3, -np.pi / 4]
            beta = [0.6, 0.8]

        fx = generate_fracture(n, m, theta, beta)
        fractures.append(fx)

    # 连接入井段
    inlet1 = [(0, 0), (0, 0.5)]
    inlet2 = [(1, 0.5), (1, 1)]
    fractures.append(inlet1)
    fractures.append(inlet2)

    return fractures


from scipy.spatial import Quadtree, Delaunay


def pebi_grid(fractures):
    # 构建四叉树
    tree = Quadtree(fractures)
    tree.construct()

    # 初始化网格
    grid = Grid()

    # 在四叉树叶子节点构建PEBI网格
    for leaf in tree.leaf_nodes:
        points = leaf.points

        # Delaunay三角剖分
        tri = Delaunay(points)

        # 检查Delaunay三角形边界
        for s in tri.simplices:
            if valid_circumcircle(points[s]):
                grid.add_cell(s)

        # 网格细化
        grid.refine(leaf.box)

    return grid


def valid_circumcircle(simplex):
    """检查Delaunay三角形边界"""
    # 判断三角形外接圆上是否有其他点
    return in_circumcircle(simplex, tri.circumcenters[simplex])


def in_circumcircle(simplex, c):
    """判断是否在外接圆内"""
    return np.linalg.norm(points[simplex] - c) ** 2 < r ** 2


import numpy as np


def transport_model(C0, grid, vel, D, adsorption):
    """示踪剂运移模型"""

    # 初始化参数
    nt, nc = grid.shape
    C = C0 * np.ones(nc)
    Ct = np.zeros((nt, nc))

    # 计算网格属性
    cell_vol, cell_faces = grid.compute_grid_props()

    # 计算转移系数
    trans_coef = compute_trans_coef(grid, vel, D)

    # 时间循环
    for t in range(nt):
        # 构建系统矩阵
        A = build_system_matrix(grid, cell_faces, trans_coef)

        # 求解线性系统
        C = np.linalg.solve(A, b)

        # 更新下一时刻浓度
        Ct[t, :] = C

        # 考虑岩石吸附作用
        C = update_adsorption(C, cell_vol, adsorption)

    return Ct


import numpy as np


# 计算转移系数
def compute_trans_coef(grid, vel, D):
    nt, nc = grid.shape
    trans_coef = np.zeros((nc, nc))

    for c1 in range(nc):
        for c2 in range(nc):
            if grid.are_neighbors(c1, c2):
                area = grid.face_area(c1, c2)
                dist = grid.cell_distance(c1, c2)
                vel_avg = (vel[c1] + vel[c2]) / 2
                trans_coef[c1, c2] = (vel_avg * area / dist +
                                      D * area / dist ** 2)

    return trans_coef


# 构建系数矩阵
def build_system_matrix(grid, cell_faces, trans_coef):
    nc = grid.num_cells
    A = np.zeros((nc, nc))

    for c1, c2 in cell_faces:
        A[c1, c1] += trans_coef[c1, c2]
        A[c2, c2] += trans_coef[c1, c2]
        A[c1, c2] -= trans_coef[c1, c2]
        A[c2, c1] -= trans_coef[c1, c2]

    return A


# 更新吸附
def update_adsorption(C, cell_vol, adsorption):
    rho_s = adsorption['rho_s']
    D_L = adsorption['D_L']

    q_e = rho_s * D_L * C
    q = adsorption['q']

    dq = (q_e - q) * dt / adsorption['tau']

    q += dq
    C_ads = q / (rho_s * cell_vol)

    return C - C_ads


import numpy as np


# 目标函数
def objective_func(x):
    n, m, theta, beta, D, adsorption = x

    # 生成裂缝网络
    fx = create_fractures(n, m, theta, beta)

    # 生成PEBI网格
    grid = pebi_grid(fx)

    # 模型预测
    C_model = transport_model(C0, grid, vel, D, adsorption)

    # 计算目标函数值
    error = np.sum((C_model - C_field) ** 2)
    return error


# 遗传算法
def genetic_optimize(C_field):
    # 遗传算法参数设置
    pop_size = 50
    mutate_rate = 0.01
    n_generations = 100

    # 创建初始群体
    pop = create_init_pop(pop_size)

    for i in range(n_generations):
        # 评价适应度
        fitness = evaluate(pop, C_field)

        # 选择
        pop = selection(pop, fitness)

        # 交叉
        pop = crossover(pop)

        # 变异
        pop = mutation(pop, mutate_rate)

    # 返回最佳个体
    best = get_best(pop, C_field)
    return best


# 主程序
if __name__ == "__main__":
    ......

    # 生成网格
    fx = create_fractures(n, m, theta, beta)
    grid = pebi_grid(fx)

    # 模型预测
    C_model = transport_model(C0, grid, vel, D, adsorption)

    # 优化参数
    x_opt = genetic_optimize(C_field)

    # 结果作图
    plot_conc(C_model)