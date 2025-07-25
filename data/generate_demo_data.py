import os  # 导入os模块，用于文件和目录操作
from PIL import Image  # 导入PIL库，用于图片生成
import cv2  # 导入OpenCV库，用于视频生成
import numpy as np  # 导入numpy库，用于生成图片和视频帧

def gen_text():
    # 生成样例文本数据
    os.makedirs('data', exist_ok=True)  # 创建data目录（如果不存在）
    with open('data/test1.txt', 'w', encoding='utf-8') as f:
        f.write('hello world')  # 写入一行文本
    with open('data/test2.txt', 'w', encoding='utf-8') as f:
        f.write('deep learning')  # 写入另一行文本

def gen_image():
    # 生成样例图片数据
    os.makedirs('data', exist_ok=True)  # 创建data目录（如果不存在）
    Image.new('RGB', (224,224), (255,0,0)).save('data/test1.jpg')  # 生成红色图片
    Image.new('RGB', (224,224), (0,255,0)).save('data/test2.jpg')  # 生成绿色图片

def gen_video():
    # 生成样例视频数据
    os.makedirs('data', exist_ok=True)  # 创建data目录（如果不存在）
    for i, color in enumerate([(255,0,0), (0,255,0)]):  # 两种颜色
        out = cv2.VideoWriter(f'data/test{i+1}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (224,224))  # 创建视频写入器
        for _ in range(2):  # 每个视频2帧
            frame = np.full((224,224,3), color, np.uint8)  # 生成纯色帧
            out.write(frame)  # 写入帧
        out.release()  # 释放资源

def main():
    gen_text()  # 生成文本数据
    gen_image()  # 生成图片数据
    gen_video()  # 生成视频数据
    print('样例数据已生成到data目录')  # 打印提示

if __name__ == '__main__':
    main()  # 如果直接运行本文件，则执行main函数 