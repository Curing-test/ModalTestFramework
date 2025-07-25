import torch  # 导入PyTorch深度学习框架
import cv2  # 导入OpenCV库，用于视频处理
from torchvision import models, transforms  # 导入torchvision的模型和预处理工具

# 定义一个视频帧分类模型的包装类
class VideoFrameModel:
    def __init__(self, model_path=None, num_classes=2):
        """
        初始化ResNet18视频帧分类模型
        model_path: 本地权重路径（可选）
        num_classes: 分类类别数
        """
        self.model = models.resnet18(pretrained=True if model_path is None else False)  # 加载预训练模型
        if model_path:
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)  # 修改输出层为指定类别数
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))  # 加载本地权重
        self.model.eval()  # 设置为推理模式
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # OpenCV帧转PIL图片
            transforms.Resize((224,224)),  # 调整大小
            transforms.ToTensor(),  # 转为张量
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # 标准化
        ])

    def predict(self, video_path, topk=2):
        """
        视频推理，取第一帧做分类，返回TopK类别索引
        video_path: 视频路径
        topk: 返回前K个类别
        """
        cap = cv2.VideoCapture(video_path)  # 打开视频文件
        ret, frame = cap.read()  # 读取第一帧
        cap.release()  # 释放视频资源
        if not ret:
            return []  # 读取失败返回空
        input_tensor = self.transform(frame).unsqueeze(0)  # 预处理并增加batch维
        with torch.no_grad():  # 关闭梯度计算，加速推理
            logits = self.model(input_tensor)  # 得到输出分数
            probs = torch.softmax(logits, dim=1)  # 转为概率
            topk_probs, topk_indices = torch.topk(probs, k=topk)  # 取概率最大的K个类别
            return topk_indices[0].tolist()  # 返回类别索引 