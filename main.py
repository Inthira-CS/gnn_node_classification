import torch
from data.generate_graph import generate_graph
from models.gcn import GCN
from training.train import train_model
from evaluation.metrics import accuracy, f1


data = generate_graph()
model = GCN(data.x.shape[1], 64, 4, 0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


model = train_model(model, data, optimizer)
model.eval()


preds = model(data).argmax(dim=1)
print("Test Accuracy:", accuracy(data.y[data.test_mask], preds[data.test_mask]))
print("Test F1:", f1(data.y[data.test_mask], preds[data.test_mask]))