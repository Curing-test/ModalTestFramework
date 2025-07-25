import torch
from PIL import Image
import cv2
import numpy as np

from models.text_model import load_text_model, infer_text, build_vocab
from models.image_model import load_image_model, infer_image
from models.video_model import load_video_model, infer_video

# 文本推理入口
# model: TextLSTMClassifier实例
# text: 输入文本
# word2idx: 词表
# 返回: top5类别索引和概率

def run_text_infer(model, text, word2idx):
    return infer_text(model, text, word2idx)

# 图片推理入口
# model: ResNet18实例
# image_path: 图片路径
# class_names: 类别名列表
# 返回: top5类别名和概率

def run_image_infer(model, image_path, class_names):
    return infer_image(model, image_path, class_names)

# 视频推理入口
# model: SimpleFrameCNN实例
# video_path: 视频路径
# class_names: 类别名列表
# 返回: 每帧top5类别名

def run_video_infer(model, video_path, class_names, frame_interval=30, max_frames=10):
    return infer_video(model, video_path, class_names, frame_interval, max_frames) 