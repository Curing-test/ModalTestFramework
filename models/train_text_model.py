import torch  # 导入PyTorch深度学习框架
from transformers import BertTokenizer, BertForSequenceClassification, AdamW  # 导入BERT相关工具和优化器
from torch.utils.data import Dataset, DataLoader  # 导入数据集和数据加载器
from sklearn.model_selection import train_test_split  # 导入数据集划分工具

# 定义自定义文本数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=32):
        self.texts = texts  # 文本列表
        self.labels = labels  # 标签列表
        self.tokenizer = tokenizer  # 分词器
        self.max_len = max_len  # 最大长度
    def __len__(self):
        return len(self.texts)  # 返回样本数
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')  # 分词编码
        item = {k: v.squeeze(0) for k, v in enc.items()}  # 去掉batch维
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # 加入标签
        return item  # 返回一个样本

def train():
    # 示例数据
    texts = ['hello world', 'deep learning', 'test sentence', 'bert model']  # 文本内容
    labels = [0, 1, 0, 1]  # 标签
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 加载分词器
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)  # 加载BERT模型
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)  # 划分训练和验证集
    train_ds = TextDataset(train_texts, train_labels, tokenizer)  # 构建训练集
    val_ds = TextDataset(val_texts, val_labels, tokenizer)  # 构建验证集
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)  # 训练集加载器
    val_loader = DataLoader(val_ds, batch_size=2)  # 验证集加载器
    optimizer = AdamW(model.parameters(), lr=2e-5)  # 优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    model.to(device)  # 模型转到设备
    for epoch in range(2):  # 训练2轮
        model.train()  # 设置为训练模式
        for batch in train_loader:  # 遍历每个batch
            batch = {k: v.to(device) for k, v in batch.items()}  # 数据转到设备
            outputs = model(**batch)  # 前向传播
            loss = outputs.loss  # 取损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新
            optimizer.zero_grad()  # 梯度清零
        print(f'Epoch {epoch+1} finished')  # 打印训练进度
    torch.save(model.state_dict(), 'models/bert_text_model.pt')  # 保存模型权重
    print('模型已保存到models/bert_text_model.pt')  # 打印提示

if __name__ == '__main__':
    train()  # 如果直接运行本文件，则执行train函数 