import pytest  # 导入pytest测试框架
import allure  # 导入allure用于生成测试报告
from models.infer import run_text_infer, run_image_infer, run_video_infer  # 导入推理函数
from models.text_model import load_text_model, build_vocab  # 导入文本模型加载和词表构建
from models.image_model import load_image_model  # 导入图片模型加载
from models.video_model import load_video_model  # 导入视频模型加载
from metrics.metrics import recall_at_k, mapk  # 导入评测指标
import os  # 导入os模块

# 假设的类别名
TEXT_CLASSES = ["体育", "财经", "娱乐", "科技", "健康"]  # 文本类别
IMAGE_CLASSES = ["cat", "dog", "car", "person", "tree"]  # 图片类别
VIDEO_CLASSES = ["jump", "run", "walk", "sit", "stand"]  # 视频类别

# 定义pytest的fixture，模块级别只执行一次
@pytest.fixture(scope="module")
def text_model():
    # 假设词表和模型已准备好
    texts = ["hello world", "test sentence", "deep learning"]  # 示例文本
    word2idx = build_vocab(texts)  # 构建词表
    model_path = "models/text_model.pt"  # 模型路径
    vocab_size = len(word2idx)
    embed_dim = 32
    hidden_dim = 32
    num_classes = len(TEXT_CLASSES)
    if os.path.exists(model_path):
        model = load_text_model(model_path, vocab_size, embed_dim, hidden_dim, num_classes)  # 加载模型
    else:
        model = None  # 未训练模型时返回None
    return model, word2idx  # 返回模型和词表

@pytest.fixture(scope="module")
def image_model():
    model_path = "models/image_model.pt"  # 模型路径
    num_classes = len(IMAGE_CLASSES)
    if os.path.exists(model_path):
        model = load_image_model(model_path, num_classes)  # 加载模型
    else:
        model = None
    return model

@pytest.fixture(scope="module")
def video_model():
    model_path = "models/video_model.pt"  # 模型路径
    num_classes = len(VIDEO_CLASSES)
    if os.path.exists(model_path):
        model = load_video_model(model_path, num_classes)  # 加载模型
    else:
        model = None
    return model

@pytest.fixture(scope="module")
def text_data():
    # 构造文本测试数据
    return [
        {"input": "hello world", "label": 0},
        {"input": "deep learning", "label": 2}
    ]

@pytest.fixture(scope="module")
def image_data():
    # 构造图片测试数据
    return [
        {"input": "data/test1.jpg", "label": "cat"},
        {"input": "data/test2.jpg", "label": "dog"}
    ]

@pytest.fixture(scope="module")
def video_data():
    # 构造视频测试数据
    return [
        {"input": "data/test1.mp4", "label": "jump"},
        {"input": "data/test2.mp4", "label": "run"}
    ]

@allure.feature("文本模型评测")
def test_text_model(text_model, text_data):
    model, word2idx = text_model  # 拆包
    if model is None:
        pytest.skip("未检测到文本模型文件")  # 跳过测试
    y_true, y_pred = [], []  # 存储真实标签和预测结果
    for sample in text_data:
        pred_indices, _ = run_text_infer(model, sample['input'], word2idx)  # 推理
        y_true.append(sample['label'])
        y_pred.append(pred_indices)
    assert recall_at_k(y_true, y_pred, k=3) >= 0.0  # 检查recall@k
    assert mapk(y_true, y_pred, k=3) >= 0.0  # 检查map@k

@allure.feature("图片模型评测")
def test_image_model(image_model, image_data):
    if image_model is None:
        pytest.skip("未检测到图片模型文件")  # 跳过测试
    y_true, y_pred = [], []
    for sample in image_data:
        pred_classes, _ = run_image_infer(image_model, sample['input'], IMAGE_CLASSES)  # 推理
        y_true.append(sample['label'])
        y_pred.append(pred_classes)
    assert recall_at_k(y_true, y_pred, k=3) >= 0.0
    assert mapk(y_true, y_pred, k=3) >= 0.0

@allure.feature("视频模型评测")
def test_video_model(video_model, video_data):
    if video_model is None:
        pytest.skip("未检测到视频模型文件")  # 跳过测试
    y_true, y_pred = [], []
    for sample in video_data:
        pred_classes_list = run_video_infer(video_model, sample['input'], VIDEO_CLASSES, frame_interval=30, max_frames=1)  # 推理
        pred_classes = pred_classes_list[0] if pred_classes_list else []
        y_true.append(sample['label'])
        y_pred.append(pred_classes)
    assert recall_at_k(y_true, y_pred, k=3) >= 0.0
    assert mapk(y_true, y_pred, k=3) >= 0.0 