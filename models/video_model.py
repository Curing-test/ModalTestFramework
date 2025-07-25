import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
from PIL import Image

class SimpleFrameCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def load_video_model(model_path, num_classes):
    model = SimpleFrameCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)

def infer_video(model, video_path, class_names, frame_interval=30, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_count = 0
    while cap.isOpened() and len(results) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            input_tensor = preprocess_frame(frame)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)
                topk = torch.topk(probs, k=5)
                results.append([class_names[i] for i in topk.indices[0].tolist()])
        frame_count += 1
    cap.release()
    return results  # 返回每帧的Top5类别名 