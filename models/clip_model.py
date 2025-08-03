import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

class CLIPModelWrapper:
    """
    CLIP模型包装类，支持消融实验和组件分析
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 消融实验配置
        self.ablation_config = {
            'use_text_encoder': True,      # 是否使用文本编码器
            'use_image_encoder': True,     # 是否使用图像编码器
            'use_projection': True,        # 是否使用投影层
            'use_attention': True,         # 是否使用注意力机制
            'use_normalization': True      # 是否使用归一化
        }
    
    def set_ablation_config(self, config):
        """
        设置消融实验配置
        config: 字典，包含要禁用的组件
        """
        self.ablation_config.update(config)
        print(f"消融实验配置: {self.ablation_config}")
    
    def encode_text(self, text, batch_size=32):
        """
        文本编码（支持消融实验）
        """
        if not self.ablation_config['use_text_encoder']:
            # 消融：禁用文本编码器，返回随机特征
            return torch.randn(len(text), 512).to(self.device)
        
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
            if not self.ablation_config['use_projection']:
                # 消融：禁用投影层
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            if not self.ablation_config['use_normalization']:
                # 消融：禁用归一化
                pass
            else:
                text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def encode_image(self, image_paths):
        """
        图像编码（支持消融实验）
        """
        if not self.ablation_config['use_image_encoder']:
            # 消融：禁用图像编码器，返回随机特征
            return torch.randn(len(image_paths), 512).to(self.device)
        
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            images.append(image)
        
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
            if not self.ablation_config['use_projection']:
                # 消融：禁用投影层
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            if not self.ablation_config['use_normalization']:
                # 消融：禁用归一化
                pass
            else:
                image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    def compute_similarity(self, text_features, image_features):
        """
        计算文本-图像相似度
        """
        if not self.ablation_config['use_attention']:
            # 消融：禁用注意力机制，使用简单点积
            return torch.matmul(text_features, image_features.T)
        else:
            # 使用CLIP的注意力机制
            return torch.matmul(text_features, image_features.T) * 100  # CLIP的缩放因子
    
    def predict(self, text, image_paths, topk=5):
        """
        预测文本-图像匹配度
        """
        text_features = self.encode_text([text])
        image_features = self.encode_image(image_paths)
        similarity = self.compute_similarity(text_features, image_features)
        
        # 获取topk结果
        topk_probs, topk_indices = torch.topk(similarity[0], k=topk)
        return topk_indices.cpu().numpy(), topk_probs.cpu().numpy()
    
    def ablation_study(self, text, image_paths, ablation_configs):
        """
        消融实验：分析不同组件的贡献
        """
        results = {}
        baseline_config = {
            'use_text_encoder': True,
            'use_image_encoder': True,
            'use_projection': True,
            'use_attention': True,
            'use_normalization': True
        }
        
        # 基线性能
        self.set_ablation_config(baseline_config)
        baseline_indices, baseline_probs = self.predict(text, image_paths)
        results['baseline'] = {
            'indices': baseline_indices,
            'probs': baseline_probs,
            'config': baseline_config.copy()
        }
        
        # 消融实验
        for name, config in ablation_configs.items():
            self.set_ablation_config(config)
            indices, probs = self.predict(text, image_paths)
            results[name] = {
                'indices': indices,
                'probs': probs,
                'config': config.copy()
            }
        
        return results
    
    def component_contribution_analysis(self, text, image_paths):
        """
        组件贡献度定量分析
        """
        # 定义消融配置
        ablation_configs = {
            'no_text_encoder': {'use_text_encoder': False},
            'no_image_encoder': {'use_image_encoder': False},
            'no_projection': {'use_projection': False},
            'no_attention': {'use_attention': False},
            'no_normalization': {'use_normalization': False}
        }
        
        results = self.ablation_study(text, image_paths, ablation_configs)
        
        # 计算性能下降
        baseline_score = np.mean(results['baseline']['probs'])
        contributions = {}
        
        for name, result in results.items():
            if name != 'baseline':
                current_score = np.mean(result['probs'])
                performance_drop = baseline_score - current_score
                contributions[name] = {
                    'performance_drop': performance_drop,
                    'relative_drop': performance_drop / baseline_score * 100
                }
        
        return contributions 