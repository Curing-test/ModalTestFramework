# 一键多模态自动化评测Makefile

.PHONY: all data clean_data train_text train_image train_video eval dist_eval test export deploy

all: data clean_data train_text train_image train_video test

# 生成样例数据
# 用法: make data

data:
	python data/generate_demo_data.py

# 数据清洗
clean_data:
	python data/clean_data.py

# 训练文本模型
train_text:
	python models/train_text_model.py

# 训练图片模型
train_image:
	python models/train_image_model.py

# 训练视频模型
train_video:
	python models/train_video_model.py

# 单机自动化评测
# 用法: make test

test:
	python main.py

# 分布式多进程评测
# 用法: make dist_eval

dist_eval:
	python distributed_eval.py

# 模型导出（以torchscript为例）
# 用法: make export

export:
	python models/export_models.py

# 模型部署（假设有deploy.py）
# 用法: make deploy

deploy:
	python deploy.py 