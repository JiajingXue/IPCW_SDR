import torch
import copy
from Data_GP import generate_survival_data
import matplotlib.pyplot as plt

from Proposed import (GMDD_IPCW, CensoringCoxEstimator,
                      CoxNN, tensor_to_cox_df, Coxpredict_lifelines,
                      ScalarNet, sample_cov, sample_var)
from functions import conditional_dcor
from lassonet.cox import CoxPHLoss
from lifelines.utils import concordance_index as lf_concordance_index

def true_G_from_dgp(X, Y, censor_rate):
    """
    X: torch.Tensor, shape (n,p)
    Y: torch.Tensor, shape (n,) or (n,1)
    """
    X = X.float()
    Y = Y.view(-1).float()

    rate_c = torch.exp(X[:, 0:3].sum(dim=1)) * censor_rate
    G = torch.exp(-rate_c * Y)
    return G

## Successive
def train_one_direction(
    x_train, y_train, delta_train, G_hat_train,
    x_valid, y_valid, delta_valid, G_hat_valid,
    prev_nets,
    num_epochs=300,
    mu_var=5.0,
    mu_cov=50.0,
    lr=1e-3,
    weight_decay=1e-4,
    hidden=(32, 32),
    seed=42,
    print_every=50
):
    torch.manual_seed(seed)

    p = x_train.shape[1]
    net = ScalarNet(p, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_valid = float("inf")

    # 先把前面方向在 train/valid 上算出来并固定
    with torch.no_grad():
        prev_train = [f(x_train).detach() for f in prev_nets]
        prev_valid = [f(x_valid).detach() for f in prev_nets]

    train_loss_list = []
    valid_loss_list = []

    for epoch in range(num_epochs + 1):
        net.train()
        f_train = net(x_train)   # (n,)

        omega_train = GMDD_IPCW(f_train, y_train, delta_train, G_hat_train)
        var_pen_train = (sample_var(f_train, unbiased=False) - 1.0) ** 2

        if len(prev_train) == 0:
            cov_pen_train = torch.zeros((), device=f_train.device, dtype=f_train.dtype)
        else:
            cov_pen_train = sum(sample_cov(fp, f_train, unbiased=False) ** 2 for fp in prev_train)

        loss_train = omega_train + mu_var * var_pen_train + mu_cov * cov_pen_train

        opt.zero_grad()
        loss_train.backward()
        opt.step()

        train_loss_list.append(loss_train.item())

        net.eval()
        with torch.no_grad():
            f_valid = net(x_valid)

            omega_valid = GMDD_IPCW(f_valid, y_valid, delta_valid, G_hat_valid)
            var_pen_valid = (sample_var(f_valid, unbiased=False) - 1.0) ** 2

            if len(prev_valid) == 0:
                cov_pen_valid = torch.zeros((), device=f_valid.device, dtype=f_valid.dtype)
            else:
                cov_pen_valid = sum(sample_cov(fp, f_valid, unbiased=False) ** 2 for fp in prev_valid)

            loss_valid = omega_valid + mu_var * var_pen_valid + mu_cov * cov_pen_valid
            valid_loss_list.append(loss_valid.item())

        if loss_valid.item() < best_valid:
            best_valid = loss_valid.item()
            best_state = copy.deepcopy(net.state_dict())

        if epoch % print_every == 0:
            print(
                f"epoch={epoch:4d} | "
                f"train_omega={omega_train.item():.6f} | "
                f"train_var={sample_var(f_train, unbiased=False).item():.6f} | "
                f"train_cov_pen={cov_pen_train.item():.6f} | "
                f"train={loss_train.item():.6f} || "
                f"valid_omega={omega_valid.item():.6f} | "
                f"valid_var={sample_var(f_valid, unbiased=False).item():.6f} | "
                f"valid_cov_pen={cov_pen_valid.item():.6f} | "
                f"valid={loss_valid.item():.6f}"
            )

    net.load_state_dict(best_state)
    return net, train_loss_list, valid_loss_list


def fit_successive_directions(
    x_train, y_train, delta_train, G_hat_train,
    x_valid, y_valid, delta_valid, G_hat_valid,
    d=2,
    num_epochs=300,
    mu_var_list=None,
    mu_cov=50.0,
    lr=1e-3,
    weight_decay=1e-4,
    hidden=(32, 32),
    base_seed=42
):
    if mu_var_list is None:
        mu_var_list = [5.0] * d

    nets = []
    histories = []

    for j in range(d):
        net_j, tr_hist, va_hist = train_one_direction(
            x_train, y_train, delta_train, G_hat_train,
            x_valid, y_valid, delta_valid, G_hat_valid,
            prev_nets=nets,
            num_epochs=num_epochs,
            mu_var=mu_var_list[j],
            mu_cov=mu_cov,
            lr=lr,
            weight_decay=weight_decay,
            hidden=hidden,
            seed=base_seed + j
        )

        nets.append(net_j.eval())
        histories.append({"train": tr_hist, "valid": va_hist})

    return nets, histories


@torch.no_grad()
def transform_successive(nets, x):
    Z_list = []
    for net in nets:
        net.eval()
        z = net(x)
        Z_list.append(z.view(-1, 1))
    return torch.cat(Z_list, dim=1)

## Downstream
def Coxpredict(rx_train, y_train, delta_train,
               rx_valid, y_valid, delta_valid,
               num_epochs, paint):
    loss_train_list = []
    loss_valid_list = []
    Cindex_train_list = []
    Cindex_valid_list = []
    p = rx_train.shape[1]
    net = CoxNN(p, (16, 4))
    opt = torch.optim.Adam(
        net.parameters(),
        lr=1e-4,
        weight_decay=1e-6,
        betas=(0.9, 0.99),
        eps=1e-8
    )

    for epoch in range(num_epochs+1):
        net.train()
        hazard_train = net(rx_train)
        opt.zero_grad()
        loss_train = CoxPHLoss("breslow")(hazard_train, torch.stack([y_train, delta_train], dim=1))
        loss_train_list.append(loss_train.item())
        cindex_train = lf_concordance_index(
            y_train.detach().cpu().numpy(),
            -hazard_train.detach().view(-1).cpu().numpy(),
            delta_train.detach().cpu().numpy()
        )

        Cindex_train_list.append(cindex_train)

        loss_train.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            hazard_valid = net(rx_valid)
            loss_valid = CoxPHLoss("breslow")(
                hazard_valid,
                torch.stack([y_valid, delta_valid], dim=1)
            )
            loss_valid_list.append(loss_valid.item())
            cindex_valid = lf_concordance_index(
                y_valid.detach().cpu().numpy(),
                -hazard_valid.detach().view(-1).cpu().numpy(),
                delta_valid.detach().cpu().numpy()
            )
            Cindex_valid_list.append(cindex_valid)

        # if epoch % 20 == 0:
        #     print(
        #         f"epoch={epoch} | "
        #         f"train loss={loss_train.item():.6f} | "
        #         f"valid loss={loss_valid.item():.6f} | "
        #         f"train cindex={cindex_train:.4f} | "
        #         f"valid cindex={cindex_valid:.4f}"
        #     )

    if paint == True:
        loss_train_s_list = torch.Tensor(loss_train_list)
        loss_valid_s_list = torch.Tensor(loss_valid_list)
        cindex_train_s_list = torch.tensor(Cindex_train_list)
        cindex_valid_s_list = torch.tensor(Cindex_valid_list)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(loss_train_s_list, label='train', linestyle='dashdot')
        axes[0].plot(loss_valid_s_list, label='valid', linestyle='dotted')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend(prop={'size': 12})
        axes[0].grid(True, alpha=0.3)

        # 右图：C-index
        axes[1].plot(cindex_train_s_list, label='train', linestyle='dashdot')
        axes[1].plot(cindex_valid_s_list, label='valid', linestyle='dotted')
        axes[1].set_title('C-index')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('C-index')
        axes[1].legend(prop={'size': 12})
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    else:
        pass

    return net


if __name__ == "__main__":
    n, rho, p, censor_rate = 1000, 0.0, 10, .1
    n_validtest = 1000
    seed = 4210000
    model = 'Cox'
    Representation_dim = 2

    # G-hat-train
    x_n0, observe_time_n0, indicator_n0, failure_n0, censor_n0, _ \
        = generate_survival_data(1000, rho, p, 1000000, model, censor_rate)

    # train
    x, observe_time, indicator, failure, censor, _ \
        = generate_survival_data(n, rho, p, seed, model, censor_rate)
    print((n - indicator.sum()) / n)

    # valid
    x_valid, observe_time_valid, indicator_valid, failure_valid, censor_valid, eta_valid \
        = generate_survival_data(n_validtest, rho, p, 1000001, model, censor_rate)

    # test
    x_test, observe_time_test, indicator_test, failure_test, censor_test, eta_test \
        = generate_survival_data(n_validtest, rho, p, 1000002, model, censor_rate)

    g_model = CensoringCoxEstimator().fit(x_n0, indicator_n0, observe_time_n0)

    # ===== predict G_hat(Y|X) =====
    gx_train = g_model.predict_G(x, observe_time)
    gx_valid = g_model.predict_G(x_valid, observe_time_valid)

    nets, histories = fit_successive_directions(
        x, observe_time, indicator, gx_train,
        x_valid, observe_time_valid, indicator_valid, gx_valid,
        d=Representation_dim,
        num_epochs=512,
        mu_var_list=[.85, .55], # direction-wise
        mu_cov= 50.,  # large
        lr= 1e-3,
        hidden=(64, 64),
        base_seed=40
    )

    Z = transform_successive(nets, x)
    Z_valid = transform_successive(nets, x_valid)
    Z_test = transform_successive(nets, x_test)


    Pred_net = Coxpredict(Z, observe_time, indicator,
                          Z_valid, observe_time_valid, indicator_valid,
                          512, False)
    Pred_net.eval()
    with torch.no_grad():
        pred = Pred_net(Z_test)

    cph = Coxpredict_lifelines(
        Z, observe_time, indicator,
        Z_valid, observe_time_valid, indicator_valid,
        penalizer=1e-3,
        l1_ratio=0.0,  # no penalization
        show_progress=False
    )

    net1, net2 = nets # two nets (representation)

    # test
    df_test_propose = tensor_to_cox_df(Z_test, observe_time_test, indicator_test)
    risk_test_propose = cph.predict_partial_hazard(df_test_propose).values.reshape(-1)

    cindex_test = lf_concordance_index(
        df_test_propose['time'].values,
        -risk_test_propose,
        df_test_propose['event'].values
    )

    cdcor_prop = conditional_dcor(failure_test, x_test, Z_test, num_anchors=100, random_state=42)

    print("CdCor(T, X | Z) Proposed :", cdcor_prop)

    RawX = Coxpredict(x, observe_time, indicator, x_valid, observe_time_valid, indicator_valid,
                      256,False)
    RawX.eval()
    with torch.no_grad():
        pred_rawx = RawX(x_test)


    print('-' * 50)
    print('C index:')
    print('C-index by Linear Cox:', cindex_test)
    print('C index by NN Cox:', lf_concordance_index(
            observe_time_test.detach().cpu().numpy(),
            -pred.detach().view(-1).cpu().numpy(),
            indicator_test.detach().cpu().numpy()))
