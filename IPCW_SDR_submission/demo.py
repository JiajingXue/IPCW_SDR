"""Minimal demonstration of IPCW-based nonlinear sufficient dimension reduction.

Run a quick example with

    python demo.py

Use ``python demo.py --full`` for the larger settings used in the original
example script. The quick run is intended to verify installation and show the
workflow; it is not designed to reproduce every table or figure in the paper.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from lifelines.utils import concordance_index as lf_concordance_index

from Data_GP import generate_survival_data
from Proposed import (
    CensoringCoxEstimator,
    CoxNN,
    Coxpredict_lifelines,
    GMDD_IPCW,
    ScalarNet,
    sample_cov,
    sample_var,
    tensor_to_cox_df,
)
from functions import conditional_dcor


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_direction(
    x_train,
    y_train,
    delta_train,
    g_train,
    x_valid,
    y_valid,
    delta_valid,
    g_valid,
    prev_nets,
    *,
    num_epochs=50,
    mu_var=5.0,
    mu_cov=50.0,
    lr=1e-3,
    weight_decay=1e-4,
    hidden=(32, 32),
    seed=42,
    print_every=25,
):
    torch.manual_seed(seed)
    net = ScalarNet(x_train.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=weight_decay
    )

    with torch.no_grad():
        prev_train = [f(x_train).detach() for f in prev_nets]
        prev_valid = [f(x_valid).detach() for f in prev_nets]

    best_state = None
    best_valid = float("inf")
    train_history = []
    valid_history = []

    for epoch in range(num_epochs + 1):
        net.train()
        f_train = net(x_train)
        objective_train = GMDD_IPCW(f_train, y_train, delta_train, g_train)
        variance_penalty = (sample_var(f_train, unbiased=False) - 1.0) ** 2
        covariance_penalty = sum(
            (sample_cov(fp, f_train, unbiased=False) ** 2 for fp in prev_train),
            start=torch.zeros((), dtype=f_train.dtype, device=f_train.device),
        )
        loss_train = (
            objective_train
            + mu_var * variance_penalty
            + mu_cov * covariance_penalty
        )

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        train_history.append(float(loss_train.detach()))

        net.eval()
        with torch.no_grad():
            f_valid = net(x_valid)
            objective_valid = GMDD_IPCW(f_valid, y_valid, delta_valid, g_valid)
            variance_penalty_valid = (
                sample_var(f_valid, unbiased=False) - 1.0
            ) ** 2
            covariance_penalty_valid = sum(
                (sample_cov(fp, f_valid, unbiased=False) ** 2 for fp in prev_valid),
                start=torch.zeros((), dtype=f_valid.dtype, device=f_valid.device),
            )
            loss_valid = (
                objective_valid
                + mu_var * variance_penalty_valid
                + mu_cov * covariance_penalty_valid
            )
            valid_history.append(float(loss_valid))

        if float(loss_valid) < best_valid:
            best_valid = float(loss_valid)
            best_state = copy.deepcopy(net.state_dict())

        if print_every and epoch % print_every == 0:
            print(
                f"direction epoch={epoch:4d} | "
                f"train loss={float(loss_train):.6f} | "
                f"valid loss={float(loss_valid):.6f}"
            )

    if best_state is None:
        raise RuntimeError("No valid network state was obtained during training.")
    net.load_state_dict(best_state)
    return net.eval(), {"train": train_history, "valid": valid_history}


def fit_successive_directions(
    x_train,
    y_train,
    delta_train,
    g_train,
    x_valid,
    y_valid,
    delta_valid,
    g_valid,
    *,
    d=2,
    num_epochs=50,
    mu_var_list=None,
    mu_cov=50.0,
    lr=1e-3,
    weight_decay=1e-4,
    hidden=(32, 32),
    base_seed=42,
):
    if d < 1:
        raise ValueError("d must be at least 1.")
    if mu_var_list is None:
        mu_var_list = [5.0] * d
    if len(mu_var_list) != d:
        raise ValueError("mu_var_list must contain one value for each direction.")

    nets = []
    histories = []
    for j in range(d):
        print(f"\nTraining representation direction {j + 1}/{d}")
        net_j, history_j = train_one_direction(
            x_train,
            y_train,
            delta_train,
            g_train,
            x_valid,
            y_valid,
            delta_valid,
            g_valid,
            nets,
            num_epochs=num_epochs,
            mu_var=mu_var_list[j],
            mu_cov=mu_cov,
            lr=lr,
            weight_decay=weight_decay,
            hidden=hidden,
            seed=base_seed + j,
        )
        nets.append(net_j)
        histories.append(history_j)
    return nets, histories


@torch.no_grad()
def transform_successive(nets, x):
    return torch.cat([net(x).reshape(-1, 1) for net in nets], dim=1)



def cox_breslow_loss(log_risk, time, event):
    """Negative Cox partial log-likelihood with the Breslow tie method."""
    log_risk = log_risk.reshape(-1)
    time = time.reshape(-1)
    event = event.reshape(-1)
    event_times = torch.unique(time[event > 0.5])
    if event_times.numel() == 0:
        raise ValueError("At least one observed event is required for Cox fitting.")

    log_likelihood = torch.zeros((), dtype=log_risk.dtype, device=log_risk.device)
    n_events = torch.zeros((), dtype=log_risk.dtype, device=log_risk.device)
    for t in event_times:
        event_at_t = (time == t) & (event > 0.5)
        d_t = event_at_t.sum().to(log_risk.dtype)
        risk_set = time >= t
        log_likelihood = log_likelihood + log_risk[event_at_t].sum()
        log_likelihood = log_likelihood - d_t * torch.logsumexp(log_risk[risk_set], dim=0)
        n_events = n_events + d_t
    return -log_likelihood / n_events.clamp_min(1.0)

def fit_nn_cox(
    x_train,
    y_train,
    delta_train,
    x_valid,
    y_valid,
    delta_valid,
    *,
    num_epochs=100,
    seed=2026,
):
    torch.manual_seed(seed)
    net = CoxNN(x_train.shape[1], (16, 4))
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=1e-4,
        weight_decay=1e-6,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    best_state = None
    best_valid = float("inf")
    for _ in range(num_epochs + 1):
        net.train()
        risk_train = net(x_train)
        loss_train = cox_breslow_loss(
            risk_train, y_train, delta_train
        )
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            risk_valid = net(x_valid)
            loss_valid = cox_breslow_loss(
                risk_valid, y_valid, delta_valid
            )
        if float(loss_valid) < best_valid:
            best_valid = float(loss_valid)
            best_state = copy.deepcopy(net.state_dict())

    if best_state is None:
        raise RuntimeError("No valid downstream Cox network state was obtained.")
    net.load_state_dict(best_state)
    return net.eval()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the larger, slower settings from the original example.",
    )
    parser.add_argument("--output", default="demo_results.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(4210000)

    if args.full:
        n_train = n_aux = n_valid = n_test = 1000
        representation_epochs = 512
        prediction_epochs = 512
        hidden = (64, 64)
    else:
        n_train = 200
        n_aux = 300
        n_valid = n_test = 200
        representation_epochs = 40
        prediction_epochs = 80
        hidden = (32, 32)

    rho = 0.0
    p = 10
    censor_rate = 0.1
    representation_dim = 2

    x_aux, y_aux, delta_aux, _, _, _ = generate_survival_data(
        n_aux, rho, p, 1000000, "Cox", censor_rate
    )
    x_train, y_train, delta_train, _, _, _ = generate_survival_data(
        n_train, rho, p, 4210000, "Cox", censor_rate
    )
    x_valid, y_valid, delta_valid, _, _, _ = generate_survival_data(
        n_valid, rho, p, 1000001, "Cox", censor_rate
    )
    x_test, y_test, delta_test, t_test, _, _ = generate_survival_data(
        n_test, rho, p, 1000002, "Cox", censor_rate
    )

    censoring_fraction = float(1.0 - delta_train.mean())
    print(f"Observed training censoring fraction: {censoring_fraction:.3f}")

    censoring_model = CensoringCoxEstimator().fit(x_aux, delta_aux, y_aux)
    g_train = censoring_model.predict_G(x_train, y_train)
    g_valid = censoring_model.predict_G(x_valid, y_valid)

    nets, _ = fit_successive_directions(
        x_train,
        y_train,
        delta_train,
        g_train,
        x_valid,
        y_valid,
        delta_valid,
        g_valid,
        d=representation_dim,
        num_epochs=representation_epochs,
        mu_var_list=[0.85, 0.55],
        mu_cov=50.0,
        lr=1e-3,
        hidden=hidden,
        base_seed=40,
    )

    z_train = transform_successive(nets, x_train)
    z_valid = transform_successive(nets, x_valid)
    z_test = transform_successive(nets, x_test)

    linear_cox = Coxpredict_lifelines(
        z_train,
        y_train,
        delta_train,
        z_valid,
        y_valid,
        delta_valid,
        penalizer=1e-3,
        l1_ratio=0.0,
        show_progress=False,
    )
    test_df = tensor_to_cox_df(z_test, y_test, delta_test)
    linear_risk = linear_cox.predict_partial_hazard(test_df).to_numpy().reshape(-1)
    linear_cindex = float(
        lf_concordance_index(
            test_df["time"].to_numpy(),
            -linear_risk,
            test_df["event"].to_numpy(),
        )
    )

    nn_cox = fit_nn_cox(
        z_train,
        y_train,
        delta_train,
        z_valid,
        y_valid,
        delta_valid,
        num_epochs=prediction_epochs,
    )
    with torch.no_grad():
        nn_risk = nn_cox(z_test)
    nn_cindex = float(
        lf_concordance_index(
            y_test.detach().cpu().numpy(),
            -nn_risk.detach().cpu().numpy(),
            delta_test.detach().cpu().numpy(),
        )
    )

    cdcor_value = float(
        conditional_dcor(
            t_test,
            x_test,
            z_test,
            num_anchors=min(100, n_test),
            random_state=42,
        )
    )

    result_text = (
        "IPCW-SDR demonstration results\n"
        f"training sample size: {n_train}\n"
        f"observed censoring fraction: {censoring_fraction:.6f}\n"
        f"representation dimension: {representation_dim}\n"
        f"test C-index (linear Cox): {linear_cindex:.6f}\n"
        f"test C-index (neural Cox): {nn_cindex:.6f}\n"
        f"test CdCor(T, X | s_hat(X)): {cdcor_value:.6f}\n"
    )
    print("\n" + result_text)
    Path(args.output).write_text(result_text, encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
