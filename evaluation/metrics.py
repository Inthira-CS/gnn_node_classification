import torch
import numpy as np


def accuracy(preds, labels):
    """Compute accuracy score."""
    preds = torch.argmax(preds, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def precision_recall_f1(preds, labels, num_classes):
    """Compute precision, recall and F1 score."""
    preds = torch.argmax(preds, dim=1)

    precision = []
    recall = []
    f1 = []

    for cls in range(num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()

        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)

        precision.append(p)
        recall.append(r)
        f1.append(f)

    return np.mean(precision), np.mean(recall), np.mean(f1)


def classification_report(preds, labels, num_classes):
    acc = accuracy(preds, labels)
    p, r, f = precision_recall_f1(preds, labels, num_classes)

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1_score": f
    }
