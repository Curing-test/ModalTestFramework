import os
import json
import matplotlib.pyplot as plt
import numpy as np
from models.clip_model import CLIPModelWrapper
from models.adversarial_attack import MultiModalAdversarialAttack
from utils.data_loader import list_multimodal_files

class AblationRobustnessStudy:
    """
    消融实验和鲁棒性研究综合类
    """
    def __init__(self, data_dir='data', output_dir='experiments/results'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化CLIP模型
        self.clip_model = CLIPModelWrapper()
        self.attack_model = MultiModalAdversarialAttack(self.clip_model)
        
        # 获取测试数据
        self.text_files, self.image_files, self.video_files = list_multimodal_files(data_dir)
        
    def run_ablation_study(self, text="a cat sitting on a chair"):
        """
        运行消融实验
        """
        print("=== 开始消融实验 ===")
        
        # 定义消融配置
        ablation_configs = {
            'no_text_encoder': {'use_text_encoder': False},
            'no_image_encoder': {'use_image_encoder': False},
            'no_projection': {'use_projection': False},
            'no_attention': {'use_attention': False},
            'no_normalization': {'use_normalization': False}
        }
        
        # 运行消融实验
        results = self.clip_model.ablation_study(text, self.image_files, ablation_configs)
        
        # 分析组件贡献
        contributions = self.clip_model.component_contribution_analysis(text, self.image_files)
        
        # 保存结果
        self._save_ablation_results(results, contributions)
        
        # 可视化结果
        self._visualize_ablation_results(contributions)
        
        return results, contributions
    
    def run_robustness_study(self, text="a cat sitting on a chair", attack_method='fgsm'):
        """
        运行鲁棒性研究
        """
        print("=== 开始鲁棒性研究 ===")
        
        # 定义扰动范围
        epsilon_range = [0.01, 0.05, 0.1, 0.15, 0.2]
        
        # 运行鲁棒性评估
        results = self.attack_model.evaluate_robustness(
            self.image_files, text, attack_method, epsilon_range
        )
        
        # 保存结果
        self._save_robustness_results(results, attack_method)
        
        # 可视化结果
        self._visualize_robustness_results(results, attack_method)
        
        return results
    
    def run_comprehensive_study(self, text="a cat sitting on a chair"):
        """
        运行综合研究：消融实验 + 鲁棒性分析
        """
        print("=== 开始综合研究 ===")
        
        # 1. 消融实验
        ablation_results, contributions = self.run_ablation_study(text)
        
        # 2. 鲁棒性研究
        fgsm_results = self.run_robustness_study(text, 'fgsm')
        pgd_results = self.run_robustness_study(text, 'pgd')
        
        # 3. 综合分析
        comprehensive_analysis = self._comprehensive_analysis(
            contributions, fgsm_results, pgd_results
        )
        
        # 4. 生成综合报告
        self._generate_comprehensive_report(comprehensive_analysis)
        
        return comprehensive_analysis
    
    def _save_ablation_results(self, results, contributions):
        """
        保存消融实验结果
        """
        output_file = os.path.join(self.output_dir, 'ablation_results.json')
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {}
        for key, value in results.items():
            serializable_results[key] = {
                'indices': value['indices'].tolist() if hasattr(value['indices'], 'tolist') else value['indices'],
                'probs': value['probs'].tolist() if hasattr(value['probs'], 'tolist') else value['probs'],
                'config': value['config']
            }
        
        data = {
            'results': serializable_results,
            'contributions': contributions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"消融实验结果已保存到: {output_file}")
    
    def _save_robustness_results(self, results, attack_method):
        """
        保存鲁棒性研究结果
        """
        output_file = os.path.join(self.output_dir, f'robustness_results_{attack_method}.json')
        
        # 转换numpy数组为列表
        serializable_results = {}
        for key, value in results.items():
            serializable_results[key] = {
                'original_probs': value['original_probs'].tolist() if hasattr(value['original_probs'], 'tolist') else value['original_probs'],
                'success_rate': value['success_rate']
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"鲁棒性研究结果已保存到: {output_file}")
    
    def _visualize_ablation_results(self, contributions):
        """
        可视化消融实验结果
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 性能下降图
        components = list(contributions.keys())
        performance_drops = [contributions[comp]['performance_drop'] for comp in components]
        relative_drops = [contributions[comp]['relative_drop'] for comp in components]
        
        # 绝对性能下降
        ax1.bar(components, performance_drops, color='skyblue')
        ax1.set_title('组件消融对性能的影响（绝对下降）')
        ax1.set_ylabel('性能下降')
        ax1.tick_params(axis='x', rotation=45)
        
        # 相对性能下降
        ax2.bar(components, relative_drops, color='lightcoral')
        ax2.set_title('组件消融对性能的影响（相对下降%）')
        ax2.set_ylabel('相对下降 (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'ablation_visualization.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"消融实验可视化结果已保存到: {output_file}")
    
    def _visualize_robustness_results(self, results, attack_method):
        """
        可视化鲁棒性研究结果
        """
        epsilons = list(results.keys())
        success_rates = [results[eps]['success_rate'] for eps in epsilons]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, success_rates, 'o-', linewidth=2, markersize=8)
        plt.title(f'模型鲁棒性分析 - {attack_method.upper()}攻击')
        plt.xlabel('扰动大小 (ε)')
        plt.ylabel('攻击成功率')
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(self.output_dir, f'robustness_visualization_{attack_method}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"鲁棒性研究可视化结果已保存到: {output_file}")
    
    def _comprehensive_analysis(self, contributions, fgsm_results, pgd_results):
        """
        综合分析消融实验和鲁棒性研究结果
        """
        analysis = {
            'component_importance': {},
            'robustness_comparison': {},
            'recommendations': []
        }
        
        # 分析组件重要性
        sorted_contributions = sorted(contributions.items(), 
                                    key=lambda x: x[1]['relative_drop'], 
                                    reverse=True)
        
        analysis['component_importance'] = {
            'most_critical': sorted_contributions[0][0],
            'least_critical': sorted_contributions[-1][0],
            'ranking': [comp for comp, _ in sorted_contributions]
        }
        
        # 比较不同攻击方法的鲁棒性
        fgsm_avg_success = np.mean([fgsm_results[eps]['success_rate'] for eps in fgsm_results])
        pgd_avg_success = np.mean([pgd_results[eps]['success_rate'] for eps in pgd_results])
        
        analysis['robustness_comparison'] = {
            'fgsm_avg_success_rate': fgsm_avg_success,
            'pgd_avg_success_rate': pgd_avg_success,
            'more_effective_attack': 'PGD' if pgd_avg_success > fgsm_avg_success else 'FGSM'
        }
        
        # 生成建议
        if fgsm_avg_success > 0.5:
            analysis['recommendations'].append("模型对FGSM攻击较为脆弱，建议增强对抗训练")
        
        if pgd_avg_success > 0.7:
            analysis['recommendations'].append("模型对PGD攻击非常脆弱，需要重点关注鲁棒性优化")
        
        critical_component = sorted_contributions[0][0]
        analysis['recommendations'].append(f"重点关注{critical_component}组件的优化，其对性能影响最大")
        
        return analysis
    
    def _generate_comprehensive_report(self, analysis):
        """
        生成综合研究报告
        """
        report_file = os.path.join(self.output_dir, 'comprehensive_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== CLIP模型消融实验与鲁棒性分析综合报告 ===\n\n")
            
            f.write("1. 组件重要性分析\n")
            f.write(f"   最重要的组件: {analysis['component_importance']['most_critical']}\n")
            f.write(f"   最不重要的组件: {analysis['component_importance']['least_critical']}\n")
            f.write(f"   组件重要性排序: {' -> '.join(analysis['component_importance']['ranking'])}\n\n")
            
            f.write("2. 鲁棒性分析\n")
            f.write(f"   FGSM攻击平均成功率: {analysis['robustness_comparison']['fgsm_avg_success_rate']:.3f}\n")
            f.write(f"   PGD攻击平均成功率: {analysis['robustness_comparison']['pgd_avg_success_rate']:.3f}\n")
            f.write(f"   更有效的攻击方法: {analysis['robustness_comparison']['more_effective_attack']}\n\n")
            
            f.write("3. 优化建议\n")
            for i, recommendation in enumerate(analysis['recommendations'], 1):
                f.write(f"   {i}. {recommendation}\n")
        
        print(f"综合研究报告已保存到: {report_file}")

def main():
    """
    主函数：运行完整的消融实验和鲁棒性研究
    """
    # 创建实验实例
    study = AblationRobustnessStudy()
    
    # 运行综合研究
    results = study.run_comprehensive_study("a cat sitting on a chair")
    
    print("\n=== 实验完成 ===")
    print("结果文件保存在 experiments/results/ 目录下")
    print("包括：")
    print("- ablation_results.json: 消融实验结果")
    print("- robustness_results_fgsm.json: FGSM鲁棒性结果")
    print("- robustness_results_pgd.json: PGD鲁棒性结果")
    print("- ablation_visualization.png: 消融实验可视化")
    print("- robustness_visualization_*.png: 鲁棒性研究可视化")
    print("- comprehensive_report.txt: 综合研究报告")

if __name__ == '__main__':
    main() 