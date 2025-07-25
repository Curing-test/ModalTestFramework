import torch  # 导入PyTorch深度学习框架
from models.text_model import BertTextModel  # 导入自定义文本模型
from models.image_model import ResNetImageModel  # 导入自定义图片模型
from models.video_model import VideoFrameModel  # 导入自定义视频模型

# 导出文本模型为TorchScript格式
def export_text_model():
    model = BertTextModel(model_path='models/bert_text_model.pt', num_labels=2).model  # 加载文本模型
    model.eval()  # 设置为推理模式
    example = {"input_ids": torch.randint(0, 100, (1, 32)), "attention_mask": torch.ones(1, 32, dtype=torch.long)}  # 构造示例输入
    # 用torch.jit.trace将模型转为TorchScript
    traced = torch.jit.trace(lambda input_ids, attention_mask: model(input_ids=input_ids, attention_mask=attention_mask).logits, (example["input_ids"], example["attention_mask"]))
    traced.save('models/bert_text_model.ptc')  # 保存为.ptc文件
    print('文本模型已导出为TorchScript: models/bert_text_model.ptc')  # 打印提示

# 导出图片模型为TorchScript格式
def export_image_model():
    model = ResNetImageModel(model_path='models/resnet_image_model.pt', num_classes=2).model  # 加载图片模型
    model.eval()  # 设置为推理模式
    example = torch.randn(1, 3, 224, 224)  # 构造示例输入
    traced = torch.jit.trace(model, example)  # 转为TorchScript
    traced.save('models/resnet_image_model.ptc')  # 保存为.ptc文件
    print('图片模型已导出为TorchScript: models/resnet_image_model.ptc')  # 打印提示

# 导出视频模型为TorchScript格式
def export_video_model():
    model = ResNetImageModel(model_path='models/resnet_video_model.pt', num_classes=2).model  # 加载视频模型（此处用图片模型结构）
    model.eval()  # 设置为推理模式
    example = torch.randn(1, 3, 224, 224)  # 构造示例输入
    traced = torch.jit.trace(model, example)  # 转为TorchScript
    traced.save('models/resnet_video_model.ptc')  # 保存为.ptc文件
    print('视频模型已导出为TorchScript: models/resnet_video_model.ptc')  # 打印提示

# 主函数，依次导出所有模型
def main():
    export_text_model()  # 导出文本模型
    export_image_model()  # 导出图片模型
    export_video_model()  # 导出视频模型

if __name__ == '__main__':
    main()  # 如果直接运行本文件，则执行main函数 