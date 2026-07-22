import torch
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
    fit CoxPH using lifelines:
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

    # train hazards
    risk_train = cph.predict_partial_hazard(df_train).values.reshape(-1)

   
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
