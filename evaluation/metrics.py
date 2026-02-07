# evaluation/metrics.py

from sklearn.metrics import accuracy_score, f1_score as sk_f1_score

def accuracy(y_true, y_pred):
    """
    Compute accuracy score.
    Works with lists, NumPy arrays, or PyTorch tensors.
    """
    # Convert tensors to lists if needed
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.cpu().numpy()
    return accuracy_score(y_true, y_pred)

def f1_score(y_true, y_pred, average="macro"):
    """
    Compute F1 score with configurable averaging.
    Default is 'macro' for multi-class problems.
    """
    # Convert tensors to lists if needed
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.cpu().numpy()
    return sk_f1_score(y_true, y_pred, average=average)