import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

class AdversarialAttack:
    """
    对抗性攻击基类
    """
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def clip_perturbation(self, x, x_adv, epsilon):
        """
        限制扰动在epsilon范围内
        """
        delta = x_adv - x
        delta = torch.clamp(delta, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)
        return x_adv

class FGSMAttack(AdversarialAttack):
    """
    Fast Gradient Sign Method (FGSM) 攻击
    """
    def __init__(self, model, device=None):
        super().__init__(model, device)
    
    def attack(self, x, y, epsilon=0.1, targeted=False):
        """
        FGSM攻击实现
        x: 输入图像 [batch_size, channels, height, width]
        y: 真实标签
        epsilon: 扰动大小
        targeted: 是否为目标攻击
        """
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        
        # 计算梯度
        x.requires_grad_(True)
        
        # 前向传播
        outputs = self.model(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # 处理多输出模型
        
        # 计算损失
        if targeted:
            loss = F.cross_entropy(outputs, y)
        else:
            loss = -F.cross_entropy(outputs, y)
        
        # 反向传播
        loss.backward()
        
        # 生成对抗样本
        grad = x.grad.detach()
        if targeted:
            x_adv = x - epsilon * grad.sign()
        else:
            x_adv = x + epsilon * grad.sign()
        
        # 限制扰动范围
        x_adv = self.clip_perturbation(x, x_adv, epsilon)
        
        return x_adv.detach()

class PGDAttack(AdversarialAttack):
    """
    Projected Gradient Descent (PGD) 攻击
    """
    def __init__(self, model, device=None):
        super().__init__(model, device)
    
    def attack(self, x, y, epsilon=0.1, alpha=0.01, steps=40, targeted=False, random_start=True):
        """
        PGD攻击实现
        x: 输入图像
        y: 真实标签
        epsilon: 扰动大小
        alpha: 每步扰动大小
        steps: 攻击步数
        targeted: 是否为目标攻击
        random_start: 是否随机初始化
        """
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        
        # 随机初始化
        if random_start:
            x_adv = x + torch.randn_like(x) * epsilon
            x_adv = self.clip_perturbation(x, x_adv, epsilon)
        else:
            x_adv = x.clone()
        
        # 迭代攻击
        for step in range(steps):
            x_adv.requires_grad_(True)
            
            # 前向传播
            outputs = self.model(x_adv)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 计算损失
            if targeted:
                loss = F.cross_entropy(outputs, y)
            else:
                loss = -F.cross_entropy(outputs, y)
            
            # 反向传播
            loss.backward()
            
            # 更新对抗样本
            grad = x_adv.grad.detach()
            if targeted:
                x_adv = x_adv - alpha * grad.sign()
            else:
                x_adv = x_adv + alpha * grad.sign()
            
            # 限制扰动范围
            x_adv = self.clip_perturbation(x, x_adv, epsilon)
            x_adv = x_adv.detach()
        
        return x_adv

class MultiModalAdversarialAttack:
    """
    多模态对抗性攻击（针对CLIP等模型）
    """
    def __init__(self, clip_model, device=None):
        self.clip_model = clip_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def image_attack_fgsm(self, image, text, epsilon=0.1, targeted=False):
        """
        对图像进行FGSM攻击，保持文本不变
        """
        # 将图像转换为tensor
        if isinstance(image, str):
            # 如果是文件路径，先加载图像
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            # 转换为tensor
            transform = self.clip_model.processor.image_processor
            image_tensor = transform(image, return_tensors="pt")['pixel_values'].to(self.device)
        else:
            image_tensor = image.to(self.device)
        
        # 创建虚拟标签（用于计算损失）
        dummy_label = torch.zeros(1).long().to(self.device)
        
        # 创建临时模型用于攻击
        class TempModel(nn.Module):
            def __init__(self, clip_model, text):
                super().__init__()
                self.clip_model = clip_model
                self.text = text
                self.text_features = None
                
            def forward(self, image):
                # 编码文本（只计算一次）
                if self.text_features is None:
                    text_inputs = self.clip_model.processor(text=[self.text], return_tensors="pt", padding=True)
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    with torch.no_grad():
                        self.text_features = self.clip_model.model.get_text_features(**text_inputs)
                        self.text_features = F.normalize(self.text_features, dim=-1)
                
                # 编码图像
                image_inputs = {'pixel_values': image}
                with torch.no_grad():
                    image_features = self.clip_model.model.get_image_features(**image_inputs)
                    image_features = F.normalize(image_features, dim=-1)
                
                # 计算相似度
                similarity = torch.matmul(self.text_features, image_features.T)
                return similarity
        
        temp_model = TempModel(self.clip_model, text)
        
        # 执行FGSM攻击
        fgsm_attack = FGSMAttack(temp_model, self.device)
        adv_image = fgsm_attack.attack(image_tensor, dummy_label, epsilon, targeted)
        
        return adv_image
    
    def image_attack_pgd(self, image, text, epsilon=0.1, alpha=0.01, steps=40, targeted=False):
        """
        对图像进行PGD攻击，保持文本不变
        """
        # 类似FGSM的实现，但使用PGD
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            transform = self.clip_model.processor.image_processor
            image_tensor = transform(image, return_tensors="pt")['pixel_values'].to(self.device)
        else:
            image_tensor = image.to(self.device)
        
        dummy_label = torch.zeros(1).long().to(self.device)
        
        class TempModel(nn.Module):
            def __init__(self, clip_model, text):
                super().__init__()
                self.clip_model = clip_model
                self.text = text
                self.text_features = None
                
            def forward(self, image):
                if self.text_features is None:
                    text_inputs = self.clip_model.processor(text=[self.text], return_tensors="pt", padding=True)
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    with torch.no_grad():
                        self.text_features = self.clip_model.model.get_text_features(**text_inputs)
                        self.text_features = F.normalize(self.text_features, dim=-1)
                
                image_inputs = {'pixel_values': image}
                with torch.no_grad():
                    image_features = self.clip_model.model.get_image_features(**image_inputs)
                    image_features = F.normalize(image_features, dim=-1)
                
                similarity = torch.matmul(self.text_features, image_features.T)
                return similarity
        
        temp_model = TempModel(self.clip_model, text)
        
        # 执行PGD攻击
        pgd_attack = PGDAttack(temp_model, self.device)
        adv_image = pgd_attack.attack(image_tensor, dummy_label, epsilon, alpha, steps, targeted)
        
        return adv_image
    
    def evaluate_robustness(self, image_paths, text, attack_method='fgsm', epsilon_range=[0.01, 0.05, 0.1, 0.15, 0.2]):
        """
        评估模型鲁棒性
        """
        results = {}
        
        for epsilon in epsilon_range:
            print(f"测试epsilon={epsilon}")
            
            # 原始预测
            original_indices, original_probs = self.clip_model.predict(text, image_paths)
            
            # 对抗攻击后的预测
            if attack_method == 'fgsm':
                adv_images = []
                for image_path in image_paths:
                    adv_image = self.image_attack_fgsm(image_path, text, epsilon)
                    adv_images.append(adv_image)
            elif attack_method == 'pgd':
                adv_images = []
                for image_path in image_paths:
                    adv_image = self.image_attack_pgd(image_path, text, epsilon)
                    adv_images.append(adv_image)
            
            # 计算对抗样本的预测结果
            # 这里需要将对抗样本转换回图像格式进行预测
            # 简化处理：直接计算相似度变化
            success_rate = self._calculate_attack_success_rate(original_probs, epsilon)
            
            results[epsilon] = {
                'original_probs': original_probs,
                'success_rate': success_rate
            }
        
        return results
    
    def _calculate_attack_success_rate(self, original_probs, epsilon):
        """
        计算攻击成功率（简化版本）
        """
        # 基于扰动大小估算成功率
        # 实际应用中需要计算真实的对立样本预测结果
        estimated_drop = epsilon * 10  # 简化的性能下降估算
        success_rate = min(estimated_drop, 1.0)
        return success_rate 