import torch.nn as nn
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index as lf_concordance_index
from lifelines import CoxPHFitter


class CensoringCoxEstimator:
    def __init__(self):
        self.cph = CoxPHFitter()
        self.feature_cols = None
        self.baseline_times_ = None
        self.baseline_cumhaz_ = None

    def fit(self, X, delta, Y):
        """
        Fit censoring model:
            event_c = 1 - delta
        """
        X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        delta_np = delta.detach().cpu().numpy().reshape(-1)
        Y_np = Y.detach().cpu().numpy().reshape(-1) if torch.is_tensor(Y) else np.asarray(Y).reshape(-1)

        event_c = 1.0 - delta_np

        self.feature_cols = [f"x{i}" for i in range(X_np.shape[1])]
        df = pd.DataFrame(X_np, columns=self.feature_cols)
        df["time"] = Y_np
        df["event_c"] = event_c

        self.cph.fit(df, duration_col="time", event_col="event_c")

        # baseline cumulative hazard for censoring model
        bh = self.cph.baseline_cumulative_hazard_
        self.baseline_times_ = bh.index.values.astype(float)
        self.baseline_cumhaz_ = bh.values.reshape(-1).astype(float)

        return self

    def predict_G(self, X, Y, eps=1e-8):
        """
        Return G_hat(Y_i | X_i) for each row.
        """
        X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        Y_np = Y.detach().cpu().numpy().reshape(-1) if torch.is_tensor(Y) else np.asarray(Y).reshape(-1)

        df_new = pd.DataFrame(X_np, columns=self.feature_cols)

        # linear predictor exp(beta^T x)
        risk_score = self.cph.predict_partial_hazard(df_new).values.reshape(-1)

        # interpolate baseline cumulative hazard at each Y_i
        H0_y = np.interp(
            Y_np,
            self.baseline_times_,
            self.baseline_cumhaz_,
            left=0.0,
            right=self.baseline_cumhaz_[-1]
        )

        # G_hat(y|x) = exp( - H0(y) * exp(beta^T x) )
        G_hat = np.exp(-H0_y * risk_score)
        G_hat = np.clip(G_hat, eps, 1.0)

        return torch.tensor(G_hat, dtype=torch.float32)

def dCov_IPCW(fx, Y, Delta, GX, eps=0.01):
    """
    Parameters: torch.tensor
    -----------
    fx    : (n,1) or (n,)     f(X_i)
    Y     : (n,1)             Y_{i,tau}, observed time
    Delta : (n,1)             0/1, event indicator
    GX    : (n,1) or (n,)     G(Y_i|X_i), estimated G(Y|X)
    """

    fx_mat = fx
    Y = Y.view(-1, 1)
    Delta = Delta.view(-1)
    GX = GX.view(-1)

    n = Y.shape[0]

    # -- D_ij --
    sq = (Y * Y).sum(dim=1, keepdim=True)
    D2 = sq + sq.t() - 2. * (Y @ Y.t())
    D2 = torch.clamp(D2, min=0.)
    D = torch.sqrt(D2 + 0.)

    # -- W_i = Delta_i/ GX_i --
    den = torch.clamp(GX, min=eps)
    W = Delta / den
    # W = torch.clamp(W, 1., 1.)
    # W = W / W.sum()

    # -- A_ij --
    A = torch.cdist(fx_mat, fx_mat, p=2)
    #
    # if zero_diag:
    #     A.fill_diagonal_(0.0)

    # -- pre-calculation --
    U = torch.outer(W, W)
    m = W.sum()
    M = (W * W).sum()
    W2 = W * W

    # part1
    Q = U * A * D
    S0 = Q.sum()
    S1 = (W * Q.sum(dim=1)).sum()
    S2 = (W * Q.sum(dim=0)).sum()
    Sww = ((U * U) * A * D).sum()
    part1 = (m * m - M) * S0 - m * (S1 + S2) + 2. * Sww

    # part2
    A_vec = A @ W
    A2_vec = A @ W2
    D_vec = D @ W
    D2_vec = D @ W2

    part2 = (
        - 2. * (W * (m + W) * A_vec * D_vec).sum()
        + 2. * (W * A2_vec * D_vec).sum()
        + 2. * (W * A_vec * D2_vec).sum()
    )

    # part3
    part3 = (W.t() @ (A @ W)) * (W @ (D @ W))
    # print('W:',W)
    # print('A:',A)
    # print('Q:',Q)
    # print(part1,part2,part3)
    # print(part1+part2+part3)
    denom = float(n * (n - 1) * (n - 2) * (n - 3))
    return (part1 + part2 + part3) / denom

def energy_distance(fx, gamma=None, unbiased=False, eps=1e-8):
    fx = fx.view(fx.shape[0], -1)

    assert fx.shape == gamma.shape, "fx and gamma must have the same shape."
    n = fx.shape[0]

    D_xg = torch.cdist(fx, gamma, p=2)
    D_xx = torch.cdist(fx, fx, p=2)
    D_gg = torch.cdist(gamma, gamma, p=2)

    if unbiased:
        assert n >= 2, "Need at least 2 samples for unbiased energy distance."
        mask = ~torch.eye(n, dtype=torch.bool, device=fx.device)
        exg = D_xg.mean()
        exx = D_xx[mask].mean()
        egg = D_gg[mask].mean()
    else:
        exg = D_xg.mean()
        exx = D_xx.mean()
        egg = D_gg.mean()

    ed = 2.0 * exg - exx - egg
    ed = torch.clamp(ed, min=0.0)

    return ed

def weighted_u4_from_pairwise(A, K, W):
    """
    A : (n,n)  X-side pairwise matrix
    K : (n,n)  Y-side pairwise matrix
    W : (n,)   IPCW weights
    """
    n = A.shape[0]
    W = W.view(-1)

    U = torch.outer(W, W)
    m = W.sum()
    M = (W * W).sum()
    W2 = W * W

    Q = U * A * K
    S0 = Q.sum()
    S1 = (W * Q.sum(dim=1)).sum()
    S2 = (W * Q.sum(dim=0)).sum()
    Sww = ((U * U) * A * K).sum()

    part1 = (m * m - M) * S0 - m * (S1 + S2) + 2.0 * Sww

    A_vec = A @ W
    A2_vec = A @ W2
    K_vec = K @ W
    K2_vec = K @ W2

    part2 = (
        -2.0 * (W * (m + W) * A_vec * K_vec).sum()
        +2.0 * (W * A2_vec * K_vec).sum()
        +2.0 * (W * A_vec * K2_vec).sum()
    )

    part3 = (W @ (A @ W)) * (W @ (K @ W))

    denom = float(n * (n - 1) * (n - 2) * (n - 3))
    return (part1 + part2 + part3) / denom

def GMDD_IPCW(fx, Y, Delta, GX, eps=0.01, kernel="abs", sigma=None):
    """
    fx    : (n,) or (n,1) or (n,d)
    Y     : (n,) or (n,1)
    Delta : (n,) or (n,1)
    GX    : (n,) or (n,1)
    """
    dtype = fx.dtype
    device = fx.device

    Y = Y.view(-1, 1).to(dtype=dtype, device=device)
    Delta = Delta.view(-1).to(dtype=dtype, device=device)
    GX = GX.view(-1).to(dtype=dtype, device=device)

    # IPCW weights
    W = Delta / torch.clamp(GX, min=eps)

    # ----- X-side kernel A -----
    if fx.dim() == 1 or (fx.dim() == 2 and fx.shape[1] == 1):
        fx = fx.view(-1)
        A = torch.outer(fx, fx)              # A_ij = s_i s_j
    else:
        # vector representation: A_ij = <s_i, s_j>
        A = fx @ fx.t()

    # ----- Y-side kernel K -----
    if kernel == "abs":
        K = torch.abs(Y - Y.t())
    elif kernel == "rbf":
        D2 = (Y - Y.t()) ** 2
        if sigma is None:
            offdiag = D2[D2 > 0]
            sigma = torch.sqrt(torch.median(offdiag) + 1e-12)
        K = torch.exp(-D2 / (2.0 * sigma ** 2))
    elif callable(kernel):
        K = kernel(Y)
    else:
        raise ValueError("kernel must be 'abs', 'rbf', or a callable.")

    # remove diagonal because the kernel is built from distinct indices
    A = A.clone()
    K = K.clone()
    A.fill_diagonal_(0.0)
    K.fill_diagonal_(0.0)

    return weighted_u4_from_pairwise(A, K, W)


class NN(nn.Module):
    def __init__(self, p, d, hidden=(32, 32), act=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = p
        torch.manual_seed(42)
        for h in hidden:
            layers += [nn.Linear(in_dim, h), act()]
            in_dim = h
        layers += [nn.Linear(in_dim, d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) # (n,d)


## Successive

class ScalarNet(nn.Module):
    def __init__(self, p, hidden=(32, 32), act=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = p
        for h in hidden:
            layers += [nn.Linear(in_dim, h), act()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (n,)

def sample_var(z, unbiased=False):
    zc = z - z.mean()
    n = z.shape[0]
    denom = (n - 1) if unbiased else n
    return (zc * zc).sum() / denom

def sample_cov(z1, z2, unbiased=False):
    z1c = z1 - z1.mean()
    z2c = z2 - z2.mean()
    n = z1.shape[0]
    denom = (n - 1) if unbiased else n
    return (z1c * z2c).sum() / denom


### DownStream
class CoxNN(nn.Module):
    def __init__(self, p, hidden, act=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = p
        for h in hidden:
            layers += [nn.Linear(in_dim, h), act()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1) # (n,d)

def tensor_to_cox_df(Z, time, event, prefix='z'):
    # Z: (n, d)
    if torch.is_tensor(Z):
        Z = Z.detach().cpu().numpy()
    if torch.is_tensor(time):
        time = time.detach().cpu().numpy()
    if torch.is_tensor(event):
        event = event.detach().cpu().numpy()

    Z = np.asarray(Z)
    time = np.asarray(time).reshape(-1)
    event = np.asarray(event).reshape(-1)

    d = Z.shape[1]
    df = pd.DataFrame(Z, columns=[f'{prefix}{j}' for j in range(d)])
    df['time'] = time
    df['event'] = event.astype(int)
    return df

def Coxpredict_lifelines(rx_train, y_train, delta_train,
                         rx_valid=None, y_valid=None, delta_valid=None,
                         penalizer=1e-3, l1_ratio=0.0, show_progress=False):
    """
    用 lifelines 的线性 Cox 模型拟合:
        h(t|z) = h0(t) exp(beta^T z)
    """
    df_train = tensor_to_cox_df(rx_train, y_train, delta_train)

    cph = CoxPHFitter(
        penalizer=penalizer,
        l1_ratio=l1_ratio
    )
    cph.fit(
        df_train,
        duration_col='time',
        event_col='event',
        show_progress=show_progress
    )

    # train 风险分数
    risk_train = cph.predict_partial_hazard(df_train).values.reshape(-1)

    # lifelines 的 concordance_index 例子里使用的是 -partial_hazard
    cindex_train = lf_concordance_index(
        df_train['time'].values,
        -risk_train,
        df_train['event'].values
    )
    # print('lifelines train C-index:', cindex_train)

    if rx_valid is not None:
        df_valid = tensor_to_cox_df(rx_valid, y_valid, delta_valid)
        risk_valid = cph.predict_partial_hazard(df_valid).values.reshape(-1)
        cindex_valid = lf_concordance_index(
            df_valid['time'].values,
            -risk_valid,
            df_valid['event'].values
        )
        # print('lifelines valid C-index:', cindex_valid)

    return cph

if __name__ == "__main__":
    import math
    import itertools
    import torch


    def _build_y_kernel(Y, kernel="abs", sigma=None):
        Y = Y.view(-1, 1)

        if kernel == "abs":
            K = torch.abs(Y - Y.t())
        elif kernel == "rbf":
            D2 = (Y - Y.t()) ** 2
            if sigma is None:
                offdiag = D2[D2 > 0]
                sigma = torch.sqrt(torch.median(offdiag) + 1e-12)
            K = torch.exp(-D2 / (2.0 * sigma ** 2))
        elif callable(kernel):
            K = kernel(Y)
        else:
            raise ValueError("kernel must be 'abs', 'rbf', or a callable.")

        return K


    def _build_x_kernel(fx):
        # 严格手稿 d=1 时，A_ij = s_i s_j
        # 为了和你当前实现兼容，若 fx 是 (n,d)，这里用内积 <f_i, f_j>
        if fx.dim() == 1:
            fx = fx.view(-1, 1)
        elif fx.dim() == 2 and fx.shape[1] == 1:
            fx = fx.view(-1, 1)
        elif fx.dim() != 2:
            raise ValueError("fx must have shape (n,), (n,1), or (n,d).")

        return fx @ fx.t()


    def h1_u4_from_submat(A4, K4):
        """
        A4, K4: shape (4,4)
        按手稿 page 6 的 h1,s 直接实现：
          h1 = 1/12 sum_{(i,j) in I4,2} A_ij K_ij
             - 1/12 sum_{(i,j,u) in I4,3} A_ij K_ju
             + 1/24 sum_{(i,j,u,t) in I4,4} A_ij K_ut
        其中 I4,k 是 4 个点上的“有序且互异”的 k 元组集合。:contentReference[oaicite:2]{index=2}
        """
        pair_sum = A4.new_zeros(())
        triple_sum = A4.new_zeros(())
        quad_sum = A4.new_zeros(())

        idx = range(4)

        # ordered pairs
        for i in idx:
            for j in idx:
                if i != j:
                    pair_sum = pair_sum + A4[i, j] * K4[i, j]

        # ordered triples
        for i in idx:
            for j in idx:
                for u in idx:
                    if (i != j) and (i != u) and (j != u):
                        triple_sum = triple_sum + A4[i, j] * K4[j, u]

        # ordered quadruples
        for i in idx:
            for j in idx:
                for u in idx:
                    for t in idx:
                        if len({i, j, u, t}) == 4:
                            quad_sum = quad_sum + A4[i, j] * K4[u, t]

        h1 = pair_sum / 12.0 - triple_sum / 12.0 + quad_sum / 24.0
        return h1


    def Omega_IPCW_u4_explicit(fx, Y, Delta, GX, eps=0.01, kernel="abs", sigma=None):
        """
        按手稿 (3.7) 的四阶 U 统计量，直接返回 Omega_n
        也就是 'negative GMDD' 的经验版本。
        """
        if fx.dim() == 1:
            fx = fx.view(-1, 1)

        Y = Y.view(-1)
        Delta = Delta.view(-1).to(dtype=fx.dtype, device=fx.device)
        GX = GX.view(-1).to(dtype=fx.dtype, device=fx.device)

        n = fx.shape[0]
        if n < 4:
            raise ValueError("Need at least 4 observations for the 4th-order U-statistic.")

        W = Delta / torch.clamp(GX, min=eps)  # IPCW 单点权重
        A = _build_x_kernel(fx)
        K = _build_y_kernel(Y, kernel=kernel, sigma=sigma).to(dtype=fx.dtype, device=fx.device)

        total = fx.new_zeros(())
        comb_count = math.comb(n, 4)

        for idx_tuple in itertools.combinations(range(n), 4):
            ids = torch.tensor(idx_tuple, device=fx.device)

            A4 = A[ids][:, ids]
            K4 = K[ids][:, ids]
            W4 = W[ids]

            h1 = h1_u4_from_submat(A4, K4)

            # 手稿 (3.7): 每个四元组乘上 prod_t Delta_it / Ghat(Y_it | X_it)
            total = total + torch.prod(W4) * h1

        omega = total / comb_count
        return omega


    def GMDD_IPCW_u4_explicit(fx, Y, Delta, GX, eps=0.01, kernel="abs", sigma=None):
        """
        返回 'GMDD' 版本，而不是 Omega 版本。
        因为手稿里 Omega_n 是 negative GMDD 的经验版本，所以这里取负号。
        """
        return -Omega_IPCW_u4_explicit(
            fx=fx, Y=Y, Delta=Delta, GX=GX,
            eps=eps, kernel=kernel, sigma=sigma
        )


    def one_trial(n=12, d=1, seed=0, kernel="abs"):
        torch.manual_seed(seed)

        if d == 1:
            fx = torch.randn(n, 1)
        else:
            fx = torch.randn(n, d)

        Y = torch.randn(n).abs() + 0.5
        Delta = torch.bernoulli(torch.full((n,), 0.8)).to(torch.float64)
        GX = 0.2 + 0.8 * torch.rand(n)  # 保证不太接近 0

        fast = GMDD_IPCW(fx, Y, Delta, GX, kernel=kernel)
        omega_u4 = Omega_IPCW_u4_explicit(fx, Y, Delta, GX, kernel=kernel)
        gmdd_u4 = GMDD_IPCW_u4_explicit(fx, Y, Delta, GX, kernel=kernel)

        print("=" * 70)
        print(f"n={n}, d={d}, seed={seed}, kernel={kernel}")
        print("fast      =", fast.item())
        print("omega_u4  =", omega_u4.item(), "   # 手稿 (3.7) 直接版")
        print("gmdd_u4   =", gmdd_u4.item(), "   # = -omega_u4")

        print("abs(fast - omega_u4) =", abs(fast - omega_u4).item())
        print("abs(fast - gmdd_u4)  =", abs(fast - gmdd_u4).item())

        denom1 = max(abs(omega_u4.item()), 1e-12)
        denom2 = max(abs(gmdd_u4.item()), 1e-12)
        print("rel(fast vs omega_u4) =", abs(fast - omega_u4).item() / denom1)
        print("rel(fast vs gmdd_u4)  =", abs(fast - gmdd_u4).item() / denom2)


    def many_trials(num_trials=5, n=12, d=1, kernel="abs"):
        diffs_to_omega = []
        diffs_to_gmdd = []

        for seed in range(num_trials):
            torch.manual_seed(seed)

            if d == 1:
                fx = torch.randn(n, 1)
            else:
                fx = torch.randn(n, d)

            Y = torch.randn(n).abs() + 0.5
            Delta = torch.bernoulli(torch.full((n,), 0.8)).to(torch.float64)
            GX = 0.2 + 0.8 * torch.rand(n)

            fast = GMDD_IPCW(fx, Y, Delta, GX, kernel=kernel)
            omega_u4 = Omega_IPCW_u4_explicit(fx, Y, Delta, GX, kernel=kernel)
            gmdd_u4 = GMDD_IPCW_u4_explicit(fx, Y, Delta, GX, kernel=kernel)

            diffs_to_omega.append(abs(fast - omega_u4).item())
            diffs_to_gmdd.append(abs(fast - gmdd_u4).item())

        print("=" * 70)
        print(f"[summary] n={n}, d={d}, kernel={kernel}, trials={num_trials}")
        print("max abs diff to omega_u4 =", max(diffs_to_omega))
        print("max abs diff to gmdd_u4  =", max(diffs_to_gmdd))
        print("mean abs diff to omega_u4 =", sum(diffs_to_omega) / len(diffs_to_omega))
        print("mean abs diff to gmdd_u4  =", sum(diffs_to_gmdd) / len(diffs_to_gmdd))


    if __name__ == "__main__":
        # 先用 d=1 严格检查手稿公式
        one_trial(n=12, d=1, seed=0, kernel="abs")
        many_trials(num_trials=5, n=12, d=1, kernel="abs")

        # 再看 d>1 时和你当前“内积版”快算法是否一致
        one_trial(n=12, d=2, seed=0, kernel="abs")
        many_trials(num_trials=5, n=12, d=2, kernel="abs")




    # def true_G_from_dgp(X, Y, censor_rate):
    #     """
    #     True censoring survival function under your DGP:
    #         C | X ~ Exp(rate = exp(sum(X[:,0:3])) * censor_rate)
    #
    #     So
    #         G(y|x) = P(C >= y | X=x) = exp(- rate(x) * y)
    #     """
    #     X = X.float()
    #     Y = Y.view(-1).float()
    #
    #     rate_c = torch.exp(X[:, 0:3].sum(dim=1)) * censor_rate
    #     G_true = torch.exp(-rate_c * Y)
    #     return G_true
    #
    # def evaluate_Ghat(G_true, G_hat, name="dataset"):
    #     G_true = G_true.view(-1).detach().cpu()
    #     G_hat = G_hat.view(-1).detach().cpu()
    #
    #     mae = torch.mean(torch.abs(G_true - G_hat)).item()
    #     rmse = torch.sqrt(torch.mean((G_true - G_hat) ** 2)).item()
    #
    #     gt = G_true.numpy()
    #     gh = G_hat.numpy()
    #     corr = np.corrcoef(gt, gh)[0, 1]
    #
    #     print(f"[{name}]")
    #     print(f"  mean(G_true) = {G_true.mean().item():.6f}")
    #     print(f"  mean(G_hat)  = {G_hat.mean().item():.6f}")
    #     print(f"  MAE          = {mae:.6f}")
    #     print(f"  RMSE         = {rmse:.6f}")
    #     print(f"  Corr         = {corr:.6f}")
    #     print()
    #
    # def plot_Ghat_vs_true(G_true, G_hat, name="dataset", max_points=1000):
    #     G_true = G_true.view(-1).detach().cpu().numpy()
    #     G_hat = G_hat.view(-1).detach().cpu().numpy()
    #
    #     n = len(G_true)
    #     if n > max_points:
    #         idx = np.random.choice(n, size=max_points, replace=False)
    #         G_true = G_true[idx]
    #         G_hat = G_hat[idx]
    #
    #     plt.figure(figsize=(5, 5))
    #     plt.scatter(G_true, G_hat, alpha=0.5, s=15)
    #     lo = min(G_true.min(), G_hat.min())
    #     hi = max(G_true.max(), G_hat.max())
    #     plt.plot([lo, hi], [lo, hi], 'r--', lw=2)
    #     plt.xlabel("True $G(Y|X)$")
    #     plt.ylabel("Estimated $G(Y|X)$")
    #     plt.title(f"{name}: True vs Estimated censoring survival")
    #     plt.tight_layout()
    #     plt.show()
    #
    # # ===== your DGP settings =====
    # n, rho, p, censor_rate = 2000, 0.0, 10, 0.0001
    # n_validtest = 1000
    #
    # # train
    # x, observe_time, indicator, failure, censor = \
    #     generate_survival_data(n, rho, p, 42, 'Cox', censor_rate)
    #
    # # valid
    # x_valid, observe_time_valid, indicator_valid, failure_valid, censor_valid = \
    #     generate_survival_data(n_validtest, rho, p, 1000001, 'Cox', censor_rate)
    #
    # # test
    # x_test, observe_time_test, indicator_test, failure_test, censor_test = \
    #     generate_survival_data(n_validtest, rho, p, 1000002, 'Cox', censor_rate)
    #
    # # ===== fit censoring Cox model on TRAIN only =====
    # g_model = CensoringCoxEstimator().fit(x, indicator, observe_time)
    #
    # # ===== predict G_hat(Y|X) =====
    # gx_train_hat = g_model.predict_G(x, observe_time)
    # gx_valid_hat = g_model.predict_G(x_valid, observe_time_valid)
    # gx_test_hat  = g_model.predict_G(x_test, observe_time_test)
    #
    # # ===== compute true G(Y|X) from DGP =====
    # gx_train_true = true_G_from_dgp(x, observe_time, censor_rate)
    # gx_valid_true = true_G_from_dgp(x_valid, observe_time_valid, censor_rate)
    # gx_test_true  = true_G_from_dgp(x_test, observe_time_test, censor_rate)
    #
    # # ===== numerical comparison =====
    # evaluate_Ghat(gx_train_true, gx_train_hat, name="train")
    # evaluate_Ghat(gx_valid_true, gx_valid_hat, name="valid")
    # evaluate_Ghat(gx_test_true, gx_test_hat, name="test")
    #
    # # ===== scatter plots =====
    # plot_Ghat_vs_true(gx_train_true, gx_train_hat, name="train")
    # plot_Ghat_vs_true(gx_valid_true, gx_valid_hat, name="valid")
    # plot_Ghat_vs_true(gx_test_true, gx_test_hat, name="test")






    ## check with exact dCov^2 and fast dCov^2 with true G-hat
    # import numpy as np
    # import torch
    # import itertools
    # import math
    # import matplotlib.pyplot as plt
    #
    #
    # def generate_simple_censored_data(n, seed=42, censor_rate=0.5):
    #     """
    #     Simple censored DGP:
    #         X ~ Bernoulli(0.5)
    #         T = 1 + X
    #         C ~ Exp(rate = censor_rate), independent of (X,T)
    #
    #     Then full-data dCov^2(X,T) = 0.25 exactly.
    #     With true IPCW weights, censored IPCW estimator should also converge to 0.25.
    #     """
    #     rng = np.random.default_rng(seed)
    #
    #     x1 = rng.binomial(1, 0.5, size=n).astype(np.float32)
    #     T = 1.0 + x1
    #     C = rng.exponential(scale=1.0 / censor_rate, size=n).astype(np.float32)
    #
    #     Y = np.minimum(T, C).astype(np.float32)
    #     Delta = (T <= C).astype(np.float32)
    #
    #     X = torch.tensor(x1.reshape(-1, 1), dtype=torch.float32)
    #     T = torch.tensor(T.reshape(-1, 1), dtype=torch.float32)
    #     C = torch.tensor(C.reshape(-1, 1), dtype=torch.float32)
    #     Y = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)
    #     Delta = torch.tensor(Delta, dtype=torch.float32)
    #
    #     # True censoring survival function G(t) = P(C >= t) = exp(-rate * t)
    #     # Since C independent of X, G(Y|X)=G(Y)=exp(-rate * Y)
    #     G_true = torch.exp(-censor_rate * Y.view(-1))
    #
    #     return X, Y, Delta, T, C, G_true
    #
    #
    # def naive_dcov_u_stat(fx, Y, Delta, GX, eps=0.01):
    #     """
    #     Exact IPCW 4th-order U-statistic using ordered quadruples.
    #     Very slow: O(n^4). Use only for small n.
    #     """
    #     fx = fx.view(Y.shape[0], -1).float()
    #     Y = Y.view(-1, 1).float()
    #     Delta = Delta.view(-1).float()
    #     GX = GX.view(-1).float()
    #
    #     n = Y.shape[0]
    #     assert n >= 4, "Need at least 4 observations."
    #
    #     A = torch.cdist(fx, fx, p=2)   # ||f_i - f_j||
    #     D = torch.cdist(Y, Y, p=1)     # |Y_i - Y_j|
    #
    #     den = torch.clamp(GX, min=eps)
    #     W = Delta / den
    #
    #     total_sum = fx.new_tensor(0.0)
    #
    #     for i, j, q, r in itertools.permutations(range(n), 4):
    #         psi = A[i, j] * D[i, j] + A[i, j] * D[q, r] - 2.0 * A[i, j] * D[i, q]
    #         weight = W[i] * W[j] * W[q] * W[r]
    #         total_sum = total_sum + weight * psi
    #
    #     denom = float(n * (n - 1) * (n - 2) * (n - 3))
    #     return total_sum / denom
    #
    #
    # # --------------------------
    # # 1) One-shot sanity check
    # # --------------------------
    # n = 5000
    # X, Y, Delta, T, C, G_true = generate_simple_censored_data(
    #     n=n, seed=42, censor_rate=0.5
    # )
    #
    # val_fast = dCov_IPCW(X, Y, Delta, G_true)
    # print(f"[One-shot fast] n={n}, estimated IPCW dCov^2 = {val_fast.item():.6f}")
    # print("Theoretical target = 0.25")
    #
    #
    # # -----------------------------------------
    # # 2) exact-vs-fast on small n (same sample)
    # # -----------------------------------------
    # print("\n=== Exact vs Fast on censored data ===")
    # small_ns = [12, 16, 20, 24, 28]
    # reps_small = 20
    #
    # exact_means = []
    # fast_means = []
    #
    # for n_small in small_ns:
    #     exact_vals = []
    #     fast_vals = []
    #
    #     for rep in range(reps_small):
    #         X, Y, Delta, T, C, G_true = generate_simple_censored_data(
    #             n=n_small, seed=1000 + rep, censor_rate=0.5
    #         )
    #
    #         v_exact = naive_dcov_u_stat(X, Y, Delta, G_true).item()
    #         v_fast = dCov_IPCW(X, Y, Delta, G_true).item()
    #
    #         exact_vals.append(v_exact)
    #         fast_vals.append(v_fast)
    #
    #     exact_mean = float(np.mean(exact_vals))
    #     fast_mean = float(np.mean(fast_vals))
    #     exact_sd = float(np.std(exact_vals))
    #     fast_sd = float(np.std(fast_vals))
    #
    #     exact_means.append(exact_mean)
    #     fast_means.append(fast_mean)
    #
    #     print(
    #         f"n={n_small:>3d} | "
    #         f"exact mean={exact_mean:.6f} (sd={exact_sd:.6f}) | "
    #         f"fast mean={fast_mean:.6f} (sd={fast_sd:.6f}) | "
    #         f"diff={abs(exact_mean-fast_mean):.6f}"
    #     )
    #
    #
    # # -----------------------------------------
    # # 3) convergence in n using fast version
    # # -----------------------------------------
    # print("\n=== Convergence in n (fast IPCW version) ===")
    # grid_n = [50, 100, 200, 500, 1000, 2000, 5000]
    # reps = 100
    #
    # mean_vals = []
    # sd_vals = []
    #
    # for n_cur in grid_n:
    #     vals = []
    #
    #     for rep in range(reps):
    #         X, Y, Delta, T, C, G_true = generate_simple_censored_data(
    #             n=n_cur, seed=200000 + 10 * n_cur + rep, censor_rate=0.5
    #         )
    #
    #         v = dCov_IPCW(X, Y, Delta, G_true).item()
    #         vals.append(v)
    #
    #     mean_v = float(np.mean(vals))
    #     sd_v = float(np.std(vals))
    #     mean_vals.append(mean_v)
    #     sd_vals.append(sd_v)
    #
    #     print(
    #         f"n={n_cur:>5d} | mean={mean_v:.6f}, sd={sd_v:.6f}, "
    #         f"bias={mean_v - 0.25:.6f}"
    #     )
    #
    # # -----------------------------------------
    # # 4) Plot convergence
    # # -----------------------------------------
    # plt.figure(figsize=(8, 5))
    # plt.errorbar(grid_n, mean_vals, yerr=sd_vals, fmt='o-', capsize=4, label='Fast IPCW dCov^2')
    # plt.axhline(0.25, color='red', linestyle='--', label='Theoretical limit = 0.25')
    # plt.xscale('log')
    # plt.xlabel('Sample size n (log scale)')
    # plt.ylabel('Estimated IPCW dCov^2')
    # plt.title('Convergence of censored IPCW dCov^2')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # from DGP import generate_survival_data
    #
    # n = 300   # 先小一点，不然 exact U-statistic 会很慢
    # rho = 0.0
    # p = 10
    # censor_rate = 0.1
    #
    # x, observe_time, indicator, failure, censor = \
    #     generate_survival_data(n, rho, p, 42, 'Cox', censor_rate)
    #
    # model = NN(p, d=2)
    # F = model.forward(x)
    #
    # gx = estimate_Ghat_cox(x, indicator, observe_time)
    # # print(gx.size())
    # # gx = torch.as_tensor(gx, dtype=torch.float32).view(-1)
    # # print(gx)
    #
    # # 1. 包含真实预测变量的组合 (理论上 dCov 应该最大)
    # val_true = dCov_IPCW(x[:, 0:3], observe_time, indicator, gx)
    #
    # # 2. 包含完全无关变量的组合 (理论上 dCov 应该接近 0)
    # # 假设 p=10, 那么 X[:, 7:10] 与生存时间无关
    # val_noise = dCov_IPCW(x[:, 7:10], observe_time, indicator, gx)
    #
    # print(f"True Predictors dCov: {val_true.item():.4f}")
    # print(f"Noise Predictors dCov: {val_noise.item():.4f}")
    #
    # ay = distance_correlation(x[:, 0:3], observe_time)
    # by = distance_correlation(x[:, 7:10], observe_time)
    #
    # a = distance_correlation(x[:, 0:3], failure)
    # b = distance_correlation(x[:, 7:10], failure)
    # print(a,b)

# if __name__ == "__main__":
#     from DGP import generate_survival_data
#
#     n = 100
#     rho = 0.0
#     p = 10
#     censor_rate = 10.
#     x_n0, observe_time_n0, indicator_n0, failure_n0, censor_n0, R1_n0, R2_n0 \
#         = generate_survival_data(n, rho, p, 1001, 'Cox', censor_rate)
#
#     x, observe_time, indicator, failure, censor, R1, R2 \
#         = generate_survival_data(n, rho, p, 42, 'Cox', censor_rate)
#
#     model = NN(p, d=2)
#     F = model.forward(x)
#
#     gx = estimate_Ghat_cox(x_n0, indicator_n0, observe_time_n0)
#
#     print(indicator.sum())
#     print(gx)