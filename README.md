# 多模态自动化评测框架

## 项目简介
本项目为支持文本、图片、视频等多模态场景的自动化评测平台，集成PyTorch（模型推理）、OpenCV（多媒体处理）、Allure（测试报告）、CI/CD自动化、邮件通知等能力，适用于大规模多模态模型的自动化评测与持续集成。

## 主要特性
- 支持文本、图片、视频多模态数据的自动化评测
- 支持mAP@K、Recall@K等主流评测指标
- 支持PyTorch模型推理，易于扩展
- 支持Allure测试报告自动生成
- 支持CI/CD自动化闭环
- 支持评测报告自动邮件通知

## 目录结构
```
./
├── data/                # 测试数据（文本、图片、视频）
├── models/              # PyTorch模型与推理代码
├── metrics/             # 评测指标实现
├── tests/               # 自动化测试用例
├── utils/               # 工具类（邮件、预处理等）
├── reports/             # Allure报告输出
├── main.py              # 主入口，调度评测流程
├── requirements.txt     # 依赖包
└── README.md            # 项目说明
```

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 准备模型和测试数据
3. 运行主程序：`python main.py`
4. 查看`reports/allure-report/index.html`，并查收邮件

## 适用场景
- 多模态大模型评测
- 自动化CI/CD集成
- 研发、测试、算法团队协作

---

如需详细定制或扩展，详见各模块代码注释。 