import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial import distance_matrix


def calc_min(v1, v2, v3):
    values = sorted([v1, v2, v3])
    indices = [[0, -1], [-1, 0], [-1, -1]]
    idx = [v1, v2, v3].index(values[0])
    return values[0], indices[idx]


def dtw(s1, s2):
    n = len(s1)
    m = len(s2)
    INF = 1e9
    cost_mat = distance_matrix(s1.reshape(n, -1), s2.reshape(m, -1))

    dp = np.ones([n, m]) * INF
    path = [[list() for i in range(m)] for j in range(n)]
    dp[0][0] = cost_mat[0][0]

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            v1 = dp[i][j - 1] if j > 0 else INF
            v2 = dp[i - 1][j] if i > 0 else INF
            v3 = dp[i - 1][j - 1] if i > 0 and j > 0 else INF

            val, indices = calc_min(v1, v2, v3)

            dp[i][j] = val + cost_mat[i][j]
            path[i][j] = path[i + indices[0]][j + indices[1]].copy()
            path[i][j].append([i + indices[0], j + indices[1]])

    path[n - 1][m - 1].append([n - 1, m - 1])
    return dp, path


def visualize(a, b, dp, path):
    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 5], height_ratios=[5, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax4 = plt.subplot(gs[3])
    ax1.plot(a, range(len(a)), c="blue")
    ax1.invert_xaxis()
    ax4.plot(b, c="orange")
    sns.heatmap(dp, ax=ax2, cmap="Blues")
    ax2.invert_yaxis()

    x = [p[0] + 0.5 for p in path]
    y = [p[1] + 0.5 for p in path]
    ax2.plot(x, y, c="red")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(a, c="blue", label="a")
    plt.plot(b, c="orange", label="b")
    for ia, ib in path:
        _x = [ia, ib]
        _y = [a[ia], b[ib]]
        plt.plot(_x, _y, c="gray")
        plt.scatter(ia, a[ia], c="r")
    plt.legend()
