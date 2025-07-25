import torch  # 导入PyTorch深度学习框架
from torchvision import models, transforms  # 导入torchvision的模型和预处理工具
from torch.utils.data import Dataset, DataLoader  # 导入数据集和数据加载器
from PIL import Image  # 导入PIL库，用于图片处理
import os  # 导入os模块，用于文件操作

# 定义自定义图片数据集类
class ImageDataset(Dataset):
    def __init__(self, img_dir, labels_dict, transform=None):
        self.img_dir = img_dir  # 图片文件夹路径
        self.labels_dict = labels_dict  # 图片名到标签的字典
        self.transform = transform  # 图片预处理方法
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]  # 只取jpg图片
    def __len__(self):
        return len(self.imgs)  # 返回图片数量
    def __getitem__(self, idx):
        img_name = self.imgs[idx]  # 图片文件名
        img_path = os.path.join(self.img_dir, img_name)  # 拼接完整路径
        image = Image.open(img_path).convert('RGB')  # 打开图片并转为RGB
        label = self.labels_dict[img_name]  # 获取标签
        if self.transform:
            image = self.transform(image)  # 预处理图片
        return image, label  # 返回图片和标签

def train():
    # 示例标签字典
    labels_dict = {'test1.jpg': 0, 'test2.jpg': 1}  # 图片名到类别的映射
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # 调整图片大小
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # 标准化
    ])
    dataset = ImageDataset('data', labels_dict, transform)  # 构建数据集
    loader = DataLoader(dataset, batch_size=2, shuffle=True)  # 构建数据加载器
    model = models.resnet18(pretrained=True)  # 加载预训练ResNet18模型
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 修改输出层为2类
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    model.to(device)  # 模型转到设备
    for epoch in range(2):  # 训练2轮
        model.train()  # 设置为训练模式
        for images, labels in loader:  # 遍历每个batch
            images, labels = images.to(device), labels.to(device)  # 数据转到设备
            outputs = model(images)  # 前向传播
            loss = torch.nn.functional.cross_entropy(outputs, labels)  # 计算交叉熵损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新
            optimizer.zero_grad()  # 梯度清零
        print(f'Epoch {epoch+1} finished')  # 打印训练进度
    torch.save(model.state_dict(), 'models/resnet_image_model.pt')  # 保存模型权重
    print('模型已保存到models/resnet_image_model.pt')  # 打印提示

if __name__ == '__main__':
    train()  # 如果直接运行本文件，则执行train函数 