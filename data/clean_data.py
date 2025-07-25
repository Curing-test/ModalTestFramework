import os  # 导入os模块，用于文件和目录操作
from PIL import Image  # 导入PIL库，用于图片处理
import cv2  # 导入OpenCV库，用于视频处理
from models.text_model import BertTextModel  # 导入文本模型
from models.image_model import ResNetImageModel  # 导入图片模型
from models.video_model import VideoFrameModel  # 导入视频模型

def clean_text_data(text_dir):
    """
    文本去重，去除空行
    text_dir: 存放文本文件的目录
    """
    for fname in os.listdir(text_dir):  # 遍历目录下所有文件
        if fname.endswith('.txt'):  # 只处理txt文件
            path = os.path.join(text_dir, fname)  # 拼接完整路径
            with open(path, 'r', encoding='utf-8') as f:
                lines = set([line.strip() for line in f if line.strip()])  # 去除空行并去重
            with open(path, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')  # 写回去重后的内容
    print('文本数据清洗完成')  # 打印提示

def check_image_data(img_dir):
    """
    检查图片是否损坏，格式是否为jpg/png
    img_dir: 存放图片的目录
    """
    for fname in os.listdir(img_dir):  # 遍历目录下所有文件
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):  # 只处理图片文件
            path = os.path.join(img_dir, fname)
            try:
                img = Image.open(path)  # 打开图片
                img.verify()  # 检查图片是否损坏
            except Exception as e:
                print(f'损坏图片: {path}, 错误: {e}')  # 打印损坏图片信息
    print('图片数据校验完成')  # 打印提示

def check_video_data(video_dir):
    """
    检查视频是否能正常打开
    video_dir: 存放视频的目录
    """
    for fname in os.listdir(video_dir):  # 遍历目录下所有文件
        if fname.lower().endswith('.mp4'):  # 只处理mp4视频
            path = os.path.join(video_dir, fname)
            cap = cv2.VideoCapture(path)  # 打开视频
            ret, _ = cap.read()  # 读取第一帧
            cap.release()  # 释放资源
            if not ret:
                print(f'损坏视频: {path}')  # 打印损坏视频信息
    print('视频数据校验完成')  # 打印提示

def list_multimodal_files(data_dir):
    """
    遍历读取data目录下所有txt、jpg、mp4文件，并打印每种类型的文件名
    data_dir: 数据目录
    """
    text_files = []
    image_files = []
    video_files = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.txt'):
            text_files.append(fname)
        elif fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(fname)
        elif fname.lower().endswith('.mp4'):
            video_files.append(fname)
    print('文本文件:', text_files)
    print('图片文件:', image_files)
    print('视频文件:', video_files)
    return text_files, image_files, video_files

def batch_infer_multimodal(data_dir):
    """
    批量读取data目录下txt、jpg、mp4文件并送入各自模型推理，输出推理结果
    """
    # 加载模型（假设权重文件已存在）
    text_model = BertTextModel(model_path='models/bert_text_model.pt', num_labels=2)
    image_model = ResNetImageModel(model_path='models/resnet_image_model.pt', num_classes=2)
    video_model = VideoFrameModel(model_path='models/resnet_video_model.pt', num_classes=2)
    # 获取文件列表
    text_files, image_files, video_files = list_multimodal_files(data_dir)
    print('\n【文本推理结果】')
    for fname in text_files:
        path = os.path.join(data_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        pred = text_model.predict(text)
        print(f'{fname}: 输入="{text}" -> 预测类别索引={pred}')
    print('\n【图片推理结果】')
    for fname in image_files:
        path = os.path.join(data_dir, fname)
        pred = image_model.predict(path)
        print(f'{fname}: 预测类别索引={pred}')
    print('\n【视频推理结果】')
    for fname in video_files:
        path = os.path.join(data_dir, fname)
        pred = video_model.predict(path)
        print(f'{fname}: 预测类别索引={pred}')

def main():
    clean_text_data('data')  # 清洗文本数据
    check_image_data('data')  # 校验图片数据
    check_video_data('data')  # 校验视频数据
    list_multimodal_files('data')  # 遍历并打印多模态文件
    batch_infer_multimodal('data')  # 批量推理多模态数据

if __name__ == '__main__':
    main()  # 如果直接运行本文件，则执行main函数 