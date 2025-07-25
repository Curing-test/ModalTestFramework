import os  # 导入os模块，用于文件和目录操作
from PIL import Image  # 导入PIL库，用于图片处理
import cv2  # 导入OpenCV库，用于视频处理

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

def main():
    clean_text_data('data')  # 清洗文本数据
    check_image_data('data')  # 校验图片数据
    check_video_data('data')  # 校验视频数据

if __name__ == '__main__':
    main()  # 如果直接运行本文件，则执行main函数 