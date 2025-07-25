import numpy as np  # 导入numpy库，用于数值计算
from sklearn.metrics import f1_score, roc_auc_score  # 导入sklearn的评测指标

# 计算Recall@K指标，支持多标签和单标签
# y_true: 真实标签或标签列表
# y_pred: 每个样本的预测TopK列表
def recall_at_k(y_true, y_pred, k=5):
    """
    计算Recall@K指标，支持多标签和单标签
    y_true: 真实标签或标签列表
    y_pred: 每个样本的预测TopK列表
    """
    recalls = []  # 用于存储每个样本的recall
    for true, pred in zip(y_true, y_pred):  # 遍历每个样本
        if isinstance(true, list):  # 多标签情况
            recalls.append(len(set(true) & set(pred[:k])) / len(true))  # 计算交集/真实标签数
        else:  # 单标签情况
            recalls.append(int(true in pred[:k]))  # 如果真实标签在TopK中，记为1，否则为0
    return np.mean(recalls)  # 返回平均recall

# 计算单个样本的AP@K
def apk(actual, predicted, k=5):
    if len(predicted) > k:
        predicted = predicted[:k]  # 只取前K个预测
    score = 0.0  # 累计分数
    num_hits = 0.0  # 命中数
    for i, p in enumerate(predicted):  # 遍历预测
        if (isinstance(actual, list) and p in actual) or (p == actual):  # 命中
            num_hits += 1.0
            score += num_hits / (i + 1.0)  # 累加分数
    return score

# 计算所有样本的mAP@K
def mapk(y_true, y_pred, k=5):
    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)])  # 对每个样本计算AP@K后取平均

# 计算macro F1分数，适用于多分类
def f1_macro(y_true, y_pred):
    """
    计算macro F1分数，适用于多分类
    y_true: 真实标签
    y_pred: 预测标签
    """
    return f1_score(y_true, y_pred, average='macro')  # 调用sklearn的f1_score

# 计算AUC分数，适用于二分类或多标签
def auc_score(y_true, y_score):
    """
    计算AUC分数，适用于二分类或多标签
    y_true: 真实标签，如[0,1,1,0]
    y_score: 预测概率，如[0.1,0.8,0.7,0.2]
    """
    return roc_auc_score(y_true, y_score)  # 调用sklearn的roc_auc_score 