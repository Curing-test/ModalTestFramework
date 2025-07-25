import multiprocessing as mp  # 导入多进程模块
from models.infer import infer_text, infer_image, infer_video  # 导入推理函数
from models.text_model import BertTextModel  # 导入文本模型
from models.image_model import ResNetImageModel  # 导入图片模型
from models.video_model import VideoFrameModel  # 导入视频模型

def eval_text_worker(texts, model_path, result_queue):
    # 文本评测子进程
    model = BertTextModel(model_path=model_path, num_labels=2)  # 加载文本模型
    for text in texts:  # 遍历每个文本
        pred = infer_text(model, text)  # 推理
        result_queue.put((text, pred))  # 结果放入队列

def eval_image_worker(image_paths, model_path, result_queue):
    # 图片评测子进程
    model = ResNetImageModel(model_path=model_path, num_classes=2)  # 加载图片模型
    for img in image_paths:  # 遍历每张图片
        pred = infer_image(model, img)  # 推理
        result_queue.put((img, pred))  # 结果放入队列

def eval_video_worker(video_paths, model_path, result_queue):
    # 视频评测子进程
    model = VideoFrameModel(model_path=model_path, num_classes=2)  # 加载视频模型
    for vid in video_paths:  # 遍历每个视频
        pred = infer_video(model, vid)  # 推理
        result_queue.put((vid, pred))  # 结果放入队列

def main():
    # 示例数据
    texts = ['hello world', 'deep learning']  # 文本列表
    images = ['data/test1.jpg', 'data/test2.jpg']  # 图片列表
    videos = ['data/test1.mp4', 'data/test2.mp4']  # 视频列表
    result_queue = mp.Queue()  # 创建进程间通信队列
    # 启动多进程
    p1 = mp.Process(target=eval_text_worker, args=(texts, 'bert-base-chinese', result_queue))  # 文本进程
    p2 = mp.Process(target=eval_image_worker, args=(images, None, result_queue))  # 图片进程
    p3 = mp.Process(target=eval_video_worker, args=(videos, None, result_queue))  # 视频进程
    p1.start(); p2.start(); p3.start()  # 启动进程
    for _ in range(len(texts)+len(images)+len(videos)):
        print(result_queue.get())  # 依次获取所有结果
    p1.join(); p2.join(); p3.join()  # 等待所有进程结束
    print('分布式评测完成')  # 打印提示

if __name__ == '__main__':
    main()  # 如果直接运行本文件，则执行main函数 