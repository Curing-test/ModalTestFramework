import torch  # 导入PyTorch深度学习框架
from torchvision import models, transforms  # 导入torchvision的模型和预处理工具
from PIL import Image  # 导入PIL库，用于图片处理

# 定义一个ResNet图片分类模型的包装类
class ResNetImageModel:
    def __init__(self, model_path=None, num_classes=2):
        """
        初始化ResNet18图片分类模型
        model_path: 本地权重路径（可选）
        num_classes: 分类类别数
        """
        self.model = models.resnet18(pretrained=True if model_path is None else False)  # 加载预训练模型
        if model_path:
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)  # 修改输出层为指定类别数
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))  # 加载本地权重
        self.model.eval()  # 设置为推理模式
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),  # 调整图片大小为224x224
            transforms.ToTensor(),  # 转为张量
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # 标准化
        ])

    def predict(self, image_path, topk=2):
        """
        图片推理，返回TopK类别索引
        image_path: 图片路径
        topk: 返回前K个类别
        """
        image = Image.open(image_path).convert('RGB')  # 打开图片并转为RGB格式
        input_tensor = self.transform(image).unsqueeze(0)  # 预处理并增加batch维
        with torch.no_grad():  # 关闭梯度计算，加速推理
            logits = self.model(input_tensor)  # 得到输出分数
            probs = torch.softmax(logits, dim=1)  # 转为概率
            topk_probs, topk_indices = torch.topk(probs, k=topk)  # 取概率最大的K个类别
            return topk_indices[0].tolist()  # 返回类别索引 