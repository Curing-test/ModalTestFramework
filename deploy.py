from flask import Flask, request, jsonify  # 导入Flask Web框架相关模块，用于搭建API服务
import torch  # 导入PyTorch深度学习框架
from models.text_model import BertTextModel  # 导入自定义的文本模型类
from models.image_model import ResNetImageModel  # 导入自定义的图片模型类
from models.video_model import VideoFrameModel  # 导入自定义的视频模型类

app = Flask(__name__)  # 创建Flask应用实例，__name__代表当前文件名

# 加载模型（这里假设模型文件已训练好并保存在models目录下）
text_model = BertTextModel(model_path='models/bert_text_model.pt', num_labels=2)  # 加载文本分类模型
image_model = ResNetImageModel(model_path='models/resnet_image_model.pt', num_classes=2)  # 加载图片分类模型
video_model = ResNetImageModel(model_path='models/resnet_video_model.pt', num_classes=2)  # 加载视频分类模型（此处实际应为VideoFrameModel，示例用法）

# 定义文本预测API接口，POST请求
@app.route('/predict/text', methods=['POST'])
def predict_text():
    data = request.json  # 获取POST请求中的JSON数据
    text = data['text']  # 提取文本内容
    result = text_model.predict(text)  # 用文本模型进行预测，返回类别索引
    return jsonify({'result': result})  # 返回预测结果，格式为JSON

# 定义图片预测API接口，POST请求
@app.route('/predict/image', methods=['POST'])
def predict_image():
    file = request.files['file']  # 获取上传的图片文件
    file.save('tmp.jpg')  # 保存到本地临时文件
    result = image_model.predict('tmp.jpg')  # 用图片模型进行预测
    return jsonify({'result': result})  # 返回预测结果

# 定义视频预测API接口，POST请求
@app.route('/predict/video', methods=['POST'])
def predict_video():
    file = request.files['file']  # 获取上传的视频文件
    file.save('tmp.mp4')  # 保存到本地临时文件
    result = video_model.predict('tmp.mp4')  # 用视频模型进行预测
    return jsonify({'result': result})  # 返回预测结果

# 如果直接运行本文件，则启动Flask Web服务，监听所有IP的5000端口
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 启动服务 