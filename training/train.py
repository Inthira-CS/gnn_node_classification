import torch
import torch.nn.functional as F

from models.gcn import GCN
from models.gat import GAT


def train(model, data, optimizer, epochs=200):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")


def evaluate(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)

    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    return train_acc, val_acc, test_acc


def run_training(data, model_type="gcn", hidden_dim=64, lr=0.005, epochs=200):
    num_classes = int(data.y.max().item()) + 1

    if model_type == "gcn":
        model = GCN(data.num_node_features, hidden_dim, num_classes)
    else:
        model = GAT(data.num_node_features, hidden_dim, num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    train(model, data, optimizer, epochs)
    train_acc, val_acc, test_acc = evaluate(model, data)

    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc:   {val_acc:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    return model
