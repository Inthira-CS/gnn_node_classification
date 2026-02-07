import torch
from data.generate_graph import generate_graph
from models.gcn import GCN
from training.train import train_model
from evaluation.metrics import accuracy, f1_score

def main():
    # Example ground truth and predictions (quick test)
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 0, 1, 0, 1]

    acc = accuracy(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print("Accuracy:", acc)
    print("F1 Score:", f1)

    # Load graph data
    data = generate_graph()

    # Define model
    model = GCN(in_channels=data.x.shape[1], hidden_channels=64, out_channels=4, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train model
    model = train_model(model, data, optimizer)

    # Switch to evaluation mode
    print(model)
    model.eval()

    # Get predictions
    with torch.no_grad():
        preds = model(data).argmax(dim=1)

    # Evaluate on test set
    test_acc = accuracy(data.y[data.test_mask].cpu(), preds[data.test_mask].cpu())
    test_f1 = f1_score(data.y[data.test_mask].cpu(), preds[data.test_mask].cpu(), average="macro")

    print("Test Accuracy:", test_acc)
    print("Test F1:", test_f1)
if __name__ == "__main__":
    main()
