"""
生成节点网络  #DEBUG 需要做测试
"""

# %% 生成节点网络
import numpy as np
import timeit
from numba import njit

# %%
WIDTH, HEIGHT = 1000, 1000  # 设置窗口大小
N_nodes = 100
N_edges = 100
N_interpolated_points = 3
N_interpolated_distance = 0.01


# %% 定义生成平面空间网络的函数  #TODO后续改成立体物理空间的网络，或者甚至可以是高维物理空间的网络

# 定义网络节点
@njit
def create_nodes(num_nodes):
    """
    生成网络节点

    Args:
        num_nodes: int, 节点数量

    Returns:
        nodes: np.array, 节点坐标
    """
    return np.random.rand(num_nodes, 2)
    pass  # function


@njit
def create_edges(num_edges, num_nodes):
    """
    生成网络边

    Args:
        num_edges (int): 边数量
        num_nodes (int): 节点数量

    Returns:
        edges (np.array): 边的两个节点索引
    """
    return np.random.randint(0, num_nodes, size=(num_edges, 2))
    pass  # function


@njit
def interpolate_points_by_num_of_points(node1, node2, num_points):
    """
    根据每条边的插值点数生成插值点

    Args:
        node1 (tuple): 起始节点坐标
        node2 (tuple): 终止节点坐标
        num_points (int): 插值点数

    Returns:
        interpolated_points: list, 插值点坐标
    """
    interpolated_points = []
    for i in range(1, num_points + 1):
        x = node1[0] + (node2[0] - node1[0]) * i / (num_points + 1)
        y = node1[1] + (node2[1] - node1[1]) * i / (num_points + 1)
        interpolated_points.append((x, y))
    return interpolated_points
    pass  # function


# 根据每条边的插值距离生成插值点
@njit
def interpolate_points_by_distance(node1, node2, distance):
    """
    根据每条边的插值距离生成插值点

    Args:
        node1 (tuple): 起始节点坐标
        node2 (tuple): 终止节点坐标
        distance (float): 插值距离

    Returns:
        interpolated_points: list, 插值点坐标
    """
    interpolated_points = []
    length = np.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2)
    num_points = int(length / distance)
    for i in range(1, num_points + 1):
        x = node1[0] + (node2[0] - node1[0]) * i / (num_points + 1)
        y = node1[1] + (node2[1] - node1[1]) * i / (num_points + 1)
        interpolated_points.append((x, y))
    return interpolated_points
    pass  # function


# %% 示例

# #DEBUG 这个用在单独的示例
# N_nodes = 100
# N_edges = 100
# N_interpolated_points = 3
# N_interpolated_distance = 0.01

nodes = create_nodes(N_nodes)
edges = create_edges(N_edges, N_nodes)

t010 = timeit.default_timer()
interpolated_points = []
for edge in edges:
    node1 = nodes[edge[0]]
    node2 = nodes[edge[1]]
    # points = interpolate_points_by_num_of_points(node1, node2, N_interpolated_points)
    points = interpolate_points_by_distance(node1, node2, N_interpolated_distance)
    interpolated_points.append(points)
t011 = timeit.default_timer()
print('Time for interpolated_points:', t011-t010)

print("Nodes:", nodes)
print("Edges:", edges)
print("Interpolated Points:", interpolated_points)

# %%
# 缩放上述的点、边、插值点到指定的大小


@njit
def scale_coordinates(coordinates, width, height):
    """
    缩放散点到指定的长宽比例值

    Args:
        coordinates (np.array): 散点坐标
        width (int): 宽度
        height (int): 高度

    Returns:
        np.array: 缩放后的散点坐标
    """
    return np.array([coordinates[:, 0] * width, coordinates[:, 1] * height]).T
    pass  # function


# %% 示例 #FIXME 这里运行会报错
scaled_nodes = scale_coordinates(nodes, WIDTH, HEIGHT)

scaled_edges_start = scale_coordinates(edges[:, 0, :], WIDTH, HEIGHT)
scaled_edges_end = scale_coordinates(edges[:, 1, :], WIDTH, HEIGHT)
scaled_edges = np.stack((scaled_edges_start, scaled_edges_end), axis=1)

scaled_interpolated_points = scale_coordinates(interpolated_points, WIDTH, HEIGHT)

# %%
