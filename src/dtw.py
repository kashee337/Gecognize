import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numba import njit


@njit
def calc_dist(v1, v2):
    dist = 0.0
    for k1, k2 in zip(v1, v2):
        dist += (k1 - k2) * (k1 - k2)
    return dist


@njit
def calc_min(v1, v2, v3):
    values = sorted([v1, v2, v3])
    indices = [[0, -1], [-1, 0], [-1, -1]]
    idx = [v1, v2, v3].index(values[0])
    return values[0], indices[idx]


@njit(nogil=True)
def dtw(_s1, _s2):
    n = len(_s1)
    m = len(_s2)
    s1 = _s1.reshape(n, -1)
    s2 = _s2.reshape(m, -1)
    dp = np.full((n + 1, m + 1), np.inf)

    dp[0][0] = calc_dist(s1[0], s2[0])
    for i in range(n):
        for j in range(m):
            dist = calc_dist(s1[i], s2[j])
            dp[i + 1][j + 1] = dist + min(dp[i + 1, j], dp[i, j + 1], dp[i, j])

    return dp[1:, 1:]


@njit
def dtw_path(dp_mat):
    n, m = dp_mat.shape
    path = [(n - 1, m - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            _, indice = calc_min(
                dp_mat[i][j - 1], dp_mat[i - 1][j], dp_mat[i - 1][j - 1]
            )
            path.append((i + indice[0], j + indice[1]))
    return path


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
