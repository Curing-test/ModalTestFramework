import os
from models.text_model import BertTextModel
from models.image_model import ResNetImageModel
from models.video_model import VideoFrameModel

def list_multimodal_files(data_dir):
    """
    遍历读取data目录下所有txt、jpg、mp4文件，返回各自完整路径列表
    """
    text_files, image_files, video_files = [], [], []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if fname.endswith('.txt'):
            text_files.append(path)
        elif fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(path)
        elif fname.lower().endswith('.mp4'):
            video_files.append(path)
    return text_files, image_files, video_files

def batch_infer_multimodal(data_dir, text_model=None, image_model=None, video_model=None):
    """
    批量读取data目录下txt、jpg、mp4文件并送入各自模型推理，返回推理结果字典
    """
    # 如果未传入模型，则自动加载
    if text_model is None:
        text_model = BertTextModel(model_path='models/bert_text_model.pt', num_labels=2)
    if image_model is None:
        image_model = ResNetImageModel(model_path='models/resnet_image_model.pt', num_classes=2)
    if video_model is None:
        video_model = VideoFrameModel(model_path='models/resnet_video_model.pt', num_classes=2)
    text_files, image_files, video_files = list_multimodal_files(data_dir)
    text_results = {}
    for path in text_files:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        pred = text_model.predict(text)
        text_results[path] = pred
    image_results = {}
    for path in image_files:
        pred = image_model.predict(path)
        image_results[path] = pred
    video_results = {}
    for path in video_files:
        pred = video_model.predict(path)
        video_results[path] = pred
    return {
        'text': text_results,
        'image': image_results,
        'video': video_results
    } 

if __name__ == '__main__':
    test_batch_infer() 