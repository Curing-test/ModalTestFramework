import torch  # 导入PyTorch深度学习框架
from PIL import Image  # 导入PIL库，用于图片处理
import cv2  # 导入OpenCV库，用于视频处理
import numpy as np  # 导入numpy库

from models.text_model import BertTextModel  # 导入自定义文本模型
from models.image_model import ResNetImageModel  # 导入自定义图片模型
from models.video_model import VideoFrameModel  # 导入自定义视频模型

# 文本推理入口
# model: BertTextModel实例
# text: 输入文本
# 返回: topK类别索引

def infer_text(model, text):
    return model.predict(text)  # 调用模型的predict方法进行文本推理

# 图片推理入口
# model: ResNetImageModel实例
# image_path: 图片路径
# 返回: topK类别索引

def infer_image(model, image_path):
    return model.predict(image_path)  # 调用模型的predict方法进行图片推理

# 视频推理入口
# model: VideoFrameModel实例
# video_path: 视频路径
# 返回: topK类别索引

def infer_video(model, video_path):
    return model.predict(video_path)  # 调用模型的predict方法进行视频推理 