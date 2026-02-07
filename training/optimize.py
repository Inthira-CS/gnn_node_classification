import optuna
import torch
import torch.nn.functional as F

from models.gcn import GCN
from models.gat import GAT


def objective(trial, data, model_type="gcn"):
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.8)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "gcn":
        model = GCN(
            in_dim=data.num_node_features,
            hidden_dim=hidden_dim,
            out_dim=int(data.y.max().item()) + 1,
            dropout=dropout,
        )
    else:
        model = GAT(
            in_dim=data.num_node_features,
            hidden_dim=hidden_dim,
            out_dim=int(data.y.max().item()) + 1,
            dropout=dropout,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()

    return acc


def run_optimization(data, model_type="gcn", n_trials=20):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data, model_type), n_trials=n_trials)
    return study.best_params
