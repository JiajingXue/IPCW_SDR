import numpy as np
import torch

def generate_survival_data(n, rho, p, seed, model, censor_rate=0.5):
    """
    Returns:
        X1 : (n,) covariate
        T  : (n,) failure time
        C  : (n,) censoring time
        Y  : (n,) observed time = min(T, C)
        Delta : (n,) censoring indicator (1 if failure observed)
    """
    if seed is not None:
        np.random.seed(seed)

    # covariate
    Sigma = np.array([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)

    if model == 'Cox':
        s1 = (
                X[:, 0] ** 2
                + X[:, 1] ** 2
                + np.sin(X[:, 2])
                + np.cos(X[:, 3] ** 2)
                + X[:, 4]
        )

        s2 = (
                X[:, 5] ** 2
                + np.cos(X[:, 6] ** 2)
                + X[:, 7] * X[:, 8]
                + X[:, 9] ** 2
        )

        # 非线性 link g(beta^T X)
        predictor = 1.0 * s1 + 0.8 * s2

        hazards = np.exp(0.5 * predictor)
        hazards = np.clip(hazards, 1e-5, 1e+5)

        T = np.random.exponential( scale = 1.0 / hazards )

    else:
        raise ValueError("Model must be 'Cox', 'AFT', or 'PO'")

    if model == 'addition':
        C = np.random.exponential(scale=1 / (np.exp((
                X[:, 5]
                - X[:, 6]
                + X[:, 7]
                + (X[:, 8])
                + X[:, 9]
        )
        ) * censor_rate))
    else:
        C = np.random.exponential(scale=1/(np.exp(np.sum(X, axis=1)) * censor_rate))

    Y = np.minimum(T, C)
    delta = (T <= C).astype(int)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    delta = torch.tensor(delta, dtype=torch.float32)
    T = torch.tensor(T, dtype=torch.float32)
    C = torch.tensor(C, dtype=torch.float32)
    predictor = torch.tensor(predictor, dtype=torch.float32).view(-1, 1)

    return X, Y, delta, T.reshape(n, 1), C.reshape(n, 1), predictor

