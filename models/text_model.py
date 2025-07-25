import torch  # 导入PyTorch深度学习框架
from transformers import BertTokenizer, BertForSequenceClassification  # 导入BERT相关的分词器和模型

# 定义一个BERT文本分类模型的包装类
class BertTextModel:
    def __init__(self, model_path='bert-base-chinese', num_labels=2):
        """
        初始化BERT文本分类模型
        model_path: 预训练模型名或本地路径
        num_labels: 分类类别数
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_path)  # 加载分词器
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)  # 加载模型
        self.model.eval()  # 设置为推理模式，关闭dropout等

    def predict(self, text, topk=2):
        """
        文本推理，返回TopK类别索引
        text: 输入文本
        topk: 返回前K个类别
        """
        # 对输入文本进行分词和编码，转为模型需要的张量格式
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=32)
        with torch.no_grad():  # 关闭梯度计算，加速推理
            logits = self.model(**inputs).logits  # 得到输出分数
            probs = torch.softmax(logits, dim=1)  # 转为概率
            topk_probs, topk_indices = torch.topk(probs, k=topk)  # 取概率最大的K个类别
            return topk_indices[0].tolist()  # 返回类别索引 