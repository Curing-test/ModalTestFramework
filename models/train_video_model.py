import torch  # 导入PyTorch深度学习框架
from torchvision import models, transforms  # 导入torchvision的模型和预处理工具
from torch.utils.data import Dataset, DataLoader  # 导入数据集和数据加载器
import cv2  # 导入OpenCV库，用于视频处理
import os  # 导入os模块，用于文件操作
import numpy as np  # 导入numpy库，用于处理帧

# 定义自定义视频帧数据集类
class VideoFrameDataset(Dataset):
    def __init__(self, video_dir, labels_dict, transform=None):
        self.video_dir = video_dir  # 视频文件夹路径
        self.labels_dict = labels_dict  # 视频名到标签的字典
        self.transform = transform  # 帧预处理方法
        self.videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]  # 只取mp4视频
    def __len__(self):
        return len(self.videos)  # 返回视频数量
    def __getitem__(self, idx):
        video_name = self.videos[idx]  # 视频文件名
        video_path = os.path.join(self.video_dir, video_name)  # 拼接完整路径
        cap = cv2.VideoCapture(video_path)  # 打开视频
        ret, frame = cap.read()  # 读取第一帧
        cap.release()  # 释放资源
        label = self.labels_dict[video_name]  # 获取标签
        if self.transform:
            frame = self.transform(frame)  # 预处理帧
        return frame, label  # 返回帧和标签

def train():
    # 示例标签字典
    labels_dict = {'test1.mp4': 0, 'test2.mp4': 1}  # 视频名到类别的映射
    transform = transforms.Compose([
        transforms.ToPILImage(),  # OpenCV帧转PIL图片
        transforms.Resize((224,224)),  # 调整大小
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # 标准化
    ])
    dataset = VideoFrameDataset('data', labels_dict, transform)  # 构建数据集
    loader = DataLoader(dataset, batch_size=2, shuffle=True)  # 构建数据加载器
    model = models.resnet18(pretrained=True)  # 加载预训练ResNet18模型
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 修改输出层为2类
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    model.to(device)  # 模型转到设备
    for epoch in range(2):  # 训练2轮
        model.train()  # 设置为训练模式
        for frames, labels in loader:  # 遍历每个batch
            frames, labels = frames.to(device), labels.to(device)  # 数据转到设备
            outputs = model(frames)  # 前向传播
            loss = torch.nn.functional.cross_entropy(outputs, labels)  # 计算交叉熵损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新
            optimizer.zero_grad()  # 梯度清零
        print(f'Epoch {epoch+1} finished')  # 打印训练进度
    torch.save(model.state_dict(), 'models/resnet_video_model.pt')  # 保存模型权重
    print('模型已保存到models/resnet_video_model.pt')  # 打印提示

if __name__ == '__main__':
    train()  # 如果直接运行本文件，则执行train函数 