import numpy as np

def recall_at_k(y_true, y_pred, k=5):
    """
    计算Recall@K
    y_true: [真实标签]
    y_pred: [[预测排序列表], ...]
    """
    recalls = []
    for true, pred in zip(y_true, y_pred):
        recalls.append(int(true in pred[:k]))
    return np.mean(recalls)

def apk(actual, predicted, k=5):
    """
    计算单样本的AP@K
    actual: 真实标签
    predicted: 预测排序列表
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score

def mapk(y_true, y_pred, k=5):
    """
    计算mAP@K
    y_true: [真实标签]
    y_pred: [[预测排序列表], ...]
    """
    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)]) 