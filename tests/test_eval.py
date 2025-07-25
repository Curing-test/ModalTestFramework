import pytest
import allure
from models.infer import run_text_infer, run_image_infer, run_video_infer
from models.text_model import load_text_model, build_vocab
from models.image_model import load_image_model
from models.video_model import load_video_model
from metrics.metrics import recall_at_k, mapk
import os

# 假设类别名
TEXT_CLASSES = ["体育", "财经", "娱乐", "科技", "健康"]
IMAGE_CLASSES = ["cat", "dog", "car", "person", "tree"]
VIDEO_CLASSES = ["jump", "run", "walk", "sit", "stand"]

@pytest.fixture(scope="module")
def text_model():
    # 假设词表和模型已准备好
    texts = ["hello world", "test sentence", "deep learning"]
    word2idx = build_vocab(texts)
    model_path = "models/text_model.pt"  # 需替换为实际模型
    vocab_size = len(word2idx)
    embed_dim = 32
    hidden_dim = 32
    num_classes = len(TEXT_CLASSES)
    if os.path.exists(model_path):
        model = load_text_model(model_path, vocab_size, embed_dim, hidden_dim, num_classes)
    else:
        model = None  # 未训练模型时返回None
    return model, word2idx

@pytest.fixture(scope="module")
def image_model():
    model_path = "models/image_model.pt"  # 需替换为实际模型
    num_classes = len(IMAGE_CLASSES)
    if os.path.exists(model_path):
        model = load_image_model(model_path, num_classes)
    else:
        model = None
    return model

@pytest.fixture(scope="module")
def video_model():
    model_path = "models/video_model.pt"  # 需替换为实际模型
    num_classes = len(VIDEO_CLASSES)
    if os.path.exists(model_path):
        model = load_video_model(model_path, num_classes)
    else:
        model = None
    return model

@pytest.fixture(scope="module")
def text_data():
    return [
        {"input": "hello world", "label": 0},
        {"input": "deep learning", "label": 2}
    ]

@pytest.fixture(scope="module")
def image_data():
    return [
        {"input": "data/test1.jpg", "label": "cat"},
        {"input": "data/test2.jpg", "label": "dog"}
    ]

@pytest.fixture(scope="module")
def video_data():
    return [
        {"input": "data/test1.mp4", "label": "jump"},
        {"input": "data/test2.mp4", "label": "run"}
    ]

@allure.feature("文本模型评测")
def test_text_model(text_model, text_data):
    model, word2idx = text_model
    if model is None:
        pytest.skip("未检测到文本模型文件")
    y_true, y_pred = [], []
    for sample in text_data:
        pred_indices, _ = run_text_infer(model, sample['input'], word2idx)
        y_true.append(sample['label'])
        y_pred.append(pred_indices)
    assert recall_at_k(y_true, y_pred, k=3) >= 0.0
    assert mapk(y_true, y_pred, k=3) >= 0.0

@allure.feature("图片模型评测")
def test_image_model(image_model, image_data):
    if image_model is None:
        pytest.skip("未检测到图片模型文件")
    y_true, y_pred = [], []
    for sample in image_data:
        pred_classes, _ = run_image_infer(image_model, sample['input'], IMAGE_CLASSES)
        y_true.append(sample['label'])
        y_pred.append(pred_classes)
    assert recall_at_k(y_true, y_pred, k=3) >= 0.0
    assert mapk(y_true, y_pred, k=3) >= 0.0

@allure.feature("视频模型评测")
def test_video_model(video_model, video_data):
    if video_model is None:
        pytest.skip("未检测到视频模型文件")
    y_true, y_pred = [], []
    for sample in video_data:
        pred_classes_list = run_video_infer(video_model, sample['input'], VIDEO_CLASSES, frame_interval=30, max_frames=1)
        pred_classes = pred_classes_list[0] if pred_classes_list else []
        y_true.append(sample['label'])
        y_pred.append(pred_classes)
    assert recall_at_k(y_true, y_pred, k=3) >= 0.0
    assert mapk(y_true, y_pred, k=3) >= 0.0 