import torch
import networkx as nx

try:
    from torch_geometric.utils import from_networkx
except ImportError:
    from_networkx = None


def generate_graph(num_nodes=5000, num_classes=4):
    G = nx.barabasi_albert_graph(num_nodes, 5)

    # Node features
    for i in G.nodes:
        G.nodes[i]["x"] = torch.randn(16)
        G.nodes[i]["y"] = torch.randint(0, num_classes, (1,)).item()

    if from_networkx is None:
        raise ImportError("torch_geometric is not installed. Run: pip install torch-geometric")

    data = from_networkx(G)
    data.x = torch.stack([data.x[i] for i in range(num_nodes)])
    data.y = torch.tensor([data.y[i] for i in range(num_nodes)])

    # Train / Val / Test split
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
