import torch
from sortedcontainers import SortedList
import numpy as np

# dCor
def distance_correlation(A, B):
    """
    calculate Distance Correlation of A and B

    param:
    A: (n, d1) for fx
    B: (n, d2) for T

    return:
    dcor: [0, 1]
    """
    n = A.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=A.device)

    # 1. 计算成对距离矩阵
    # 使用 torch.cdist 是最快且数值稳定的方法
    dist_A = torch.cdist(A, A, p=2)
    dist_B = torch.cdist(B, B, p=2)

    # 2. 双重中心化 (Double Centering)
    # 矩阵形式: H = I - 11^T / n
    # 中心化后的矩阵: D = H * dist * H
    def double_center(dist_mat):
        row_mean = dist_mat.mean(dim=1, keepdim=True)
        col_mean = dist_mat.mean(dim=0, keepdim=True)
        all_mean = dist_mat.mean()
        return dist_mat - row_mean - col_mean + all_mean

    A_centered = double_center(dist_A)
    B_centered = double_center(dist_B)

    # 3. 计算距离协方差 dCov^2 (A, B)
    # 公式: 1/n^2 * sum(A_centered * B_centered)
    dcov2_ab = torch.sum(A_centered * B_centered) / (n * n)

    # 4. 计算距离方差 dVar^2 (A) 和 dVar^2 (B)
    dcov2_aa = torch.sum(A_centered * A_centered) / (n * n)
    dcov2_bb = torch.sum(B_centered * B_centered) / (n * n)

    # 5. 计算 dCor = sqrt( dCov^2(A,B) / sqrt(dVar^2(A) * dVar^2(B)) )
    # 加入 eps 防止分母为 0
    eps = 1e-8
    dcor2 = dcov2_ab / (torch.sqrt(dcov2_aa * dcov2_bb) + eps)

    # 保证数值在 [0, 1] 范围内
    return torch.sqrt(torch.clamp(dcor2, min=0.0))


# CdCor
def _to_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a

def _pairwise_dist(U):
    """
    U: (n, d)
    return: (n, n) Euclidean distance matrix
    """
    U = _to_2d(U)
    sq = np.sum(U * U, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * U @ U.T
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)

def _median_bandwidth(Z, eps=1e-12):
    DZ = _pairwise_dist(Z)
    vals = DZ[DZ > eps]
    if vals.size == 0:
        return 1.0
    return np.median(vals)

def _weighted_center_distance(D, w):
    """
    D: (n, n) pairwise distance matrix
    w: (n,) nonnegative weights summing to 1

    weighted double-centering:
      A_ij^w = D_ij - E_w[D_i.] - E_w[D_.j] + E_{w,w}[D]
    """
    row_mean = D @ w                     # (n,)
    grand_mean = w @ row_mean            # scalar
    Dc = D - row_mean[:, None] - row_mean[None, :] + grand_mean
    return Dc

def conditional_dcor(
    T,
    X,
    Z,
    bandwidth=None,
    bandwidth_scale=1.0,
    kernel="gaussian",
    num_anchors=100,
    random_state=0,
    min_effective_weight=1e-6,
    standardize=True,
    return_details=False,
):
    """
    Practical kernel estimator of CdCor(T, X | Z).

    Parameters
    ----------
    T : array-like, shape (n,) or (n,1)
        Failure time (prefer true T in simulation).
    X : array-like, shape (n,p)
        Original covariates.
    Z : array-like, shape (n,d)
        Learned representation, e.g. \hat s(X).
    bandwidth : float or None
        Kernel bandwidth on Z. If None, use median heuristic.
    bandwidth_scale : float
        Multiplier for the bandwidth.
    kernel : {"gaussian", "epanechnikov"}
        Kernel type.
    num_anchors : int or None
        Number of anchor points in Z used to average local dCor.
        If None or >= n, use all n points (more exact, slower).
    random_state : int
        Random seed for anchor subsampling.
    min_effective_weight : float
        Skip anchors whose local weights are too degenerate.
    standardize : bool
        Whether to standardize each column of T, X, Z.
    return_details : bool
        If True, also return local dCor values and bandwidth.

    Returns
    -------
    cdcor : float
        Estimated conditional distance correlation. Smaller is better.
    details : dict, optional
    """
    T = _to_2d(T).astype(float)
    X = _to_2d(X).astype(float)
    Z = _to_2d(Z).astype(float)

    n = T.shape[0]
    if X.shape[0] != n or Z.shape[0] != n:
        raise ValueError("T, X, Z must have the same number of rows.")

    if standardize:
        def _stdize(A):
            mu = A.mean(axis=0, keepdims=True)
            sd = A.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)
            return (A - mu) / sd

        T = _stdize(T)
        X = _stdize(X)
        Z = _stdize(Z)

    # pairwise distances for T and X
    DT = _pairwise_dist(T)   # (n,n)
    DX = _pairwise_dist(X)   # (n,n)

    # bandwidth on Z
    if bandwidth is None:
        bandwidth = _median_bandwidth(Z)
    h = max(bandwidth * bandwidth_scale, 1e-8)

    # choose anchors
    if (num_anchors is None) or (num_anchors >= n):
        anchor_idx = np.arange(n)
    else:
        rng = np.random.default_rng(random_state)
        anchor_idx = rng.choice(n, size=num_anchors, replace=False)

    local_cdcor_list = []

    for k in anchor_idx:
        dz = np.linalg.norm(Z - Z[k], axis=1)

        if kernel == "gaussian":
            wk = np.exp(-(dz ** 2) / (2.0 * h * h))
        elif kernel == "epanechnikov":
            u = dz / h
            wk = 0.75 * (1.0 - u ** 2)
            wk[u > 1.0] = 0.0
            wk = np.maximum(wk, 0.0)
        else:
            raise ValueError("kernel must be 'gaussian' or 'epanechnikov'")

        sw = wk.sum()
        if sw <= 1e-12:
            continue

        w = wk / sw

        # effective sample size under weights
        eff_n = 1.0 / np.sum(w ** 2)
        if eff_n < 5:
            continue

        # locally weighted centered distance matrices
        AT = _weighted_center_distance(DT, w)
        AX = _weighted_center_distance(DX, w)

        WW = np.outer(w, w)

        local_dcov2 = np.sum(WW * AT * AX)
        local_dvarT2 = np.sum(WW * AT * AT)
        local_dvarX2 = np.sum(WW * AX * AX)

        denom = np.sqrt(max(local_dvarT2, 0.0) * max(local_dvarX2, 0.0))

        if denom <= min_effective_weight:
            local_cdcor = 0.0
        else:
            # clip for numerical stability
            local_cdcor = local_dcov2 / denom
            local_cdcor = float(np.clip(local_cdcor, -1.0, 1.0))

        local_cdcor_list.append(local_cdcor)

    if len(local_cdcor_list) == 0:
        cdcor = np.nan
    else:
        cdcor = float(np.mean(local_cdcor_list))

    if return_details:
        return cdcor, {
            "bandwidth": h,
            "num_anchors_used": len(local_cdcor_list),
            "local_cdcor": np.array(local_cdcor_list),
        }
    return cdcor


# C-index
def concordance_index(pred, time, event):
    risk = pred

    assert len(risk) == len(time) == len(event)
    n= len(risk)
    order = sorted(range(n), key = time.__getitem__)
    past = SortedList()
    num = 0
    den = 0
    for i in order:
        num += len(past) - past.bisect_right(risk[i])
        den += len(past)
        if event[i]:
            past.add(risk[i])
    return num/den


if __name__ == "__main__":
    n = 10000
    torch.manual_seed(42)

    # 情况 1: 强线性相关
    X1 = torch.randn(n, 2)
    T1 = X1[:, 0:1] * 2.0  # T 完美由 X 的第一维决定

    # 情况 2: 强非线性相关 (圆形分布)
    X2 = torch.randn(n, 2)
    T2 = (X2[:, 0:1] ** 2 + X2[:, 1:2] ** 2).sqrt()

    # 情况 3: 完全独立
    X3 = torch.randn(n, 2)
    T3 = torch.randn(n, 1)

    print(f"线性相关 dCor: {distance_correlation(X1, T1).item():.4f}")  # 应接近 1
    print(f"非线性相关 dCor: {distance_correlation(X2, T2).item():.4f}")  # 应远大于 0
    print(f"相互独立 dCor: {distance_correlation(X3, T3).item():.4f}")  # 应接近 0
