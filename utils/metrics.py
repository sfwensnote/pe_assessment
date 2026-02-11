# @author Coder建设｜javpower
"""
评估指标模块
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import pearsonr


class AssessmentMetrics:
    """体育动作评估指标"""
    
    @staticmethod
    def compute_classification_accuracy(predictions: np.ndarray, 
                                        targets: np.ndarray) -> Dict[str, float]:
        """
        计算分类准确率
        
        Args:
            predictions: [N] 预测标签
            targets: [N] 真实标签
        """
        accuracy = np.mean(predictions == targets)
        
        # 每类准确率
        per_class_acc = {}
        for cls in np.unique(targets):
            mask = targets == cls
            if mask.sum() > 0:
                per_class_acc[f'class_{cls}'] = np.mean(predictions[mask] == targets[mask])
        
        return {
            'accuracy': float(accuracy),
            'per_class': per_class_acc
        }
    
    @staticmethod
    def compute_phase_metrics(pred_phases: np.ndarray, 
                              true_phases: np.ndarray,
                              num_phases: int) -> Dict[str, float]:
        """
        计算阶段分割指标
        
        Args:
            pred_phases: [T] 预测阶段
            true_phases: [T] 真实阶段
            num_phases: 阶段数量
        """
        # 帧级准确率
        frame_acc = np.mean(pred_phases == true_phases)
        
        # 编辑距离
        edit_dist = levenshtein_distance(pred_phases, true_phases)
        edit_score = 1 - edit_dist / len(true_phases)
        
        # 边界检测F1
        pred_boundaries = set(np.where(np.diff(pred_phases) != 0)[0])
        true_boundaries = set(np.where(np.diff(true_phases) != 0)[0])
        
        if len(pred_boundaries) == 0 and len(true_boundaries) == 0:
            boundary_f1 = 1.0
        elif len(pred_boundaries) == 0 or len(true_boundaries) == 0:
            boundary_f1 = 0.0
        else:
            tp = len(pred_boundaries & true_boundaries)
            fp = len(pred_boundaries - true_boundaries)
            fn = len(true_boundaries - pred_boundaries)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            boundary_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 每阶段准确率
        phase_acc = {}
        for i in range(num_phases):
            mask = true_phases == i
            if mask.sum() > 0:
                phase_acc[f'phase_{i}'] = np.mean(pred_phases[mask] == true_phases[mask])
        
        return {
            'frame_accuracy': float(frame_acc),
            'edit_score': float(edit_score),
            'boundary_f1': float(boundary_f1),
            'per_phase_accuracy': phase_acc
        }
    
    @staticmethod
    def compute_quality_metrics(predictions: np.ndarray, 
                                targets: np.ndarray) -> Dict[str, float]:
        """
        计算质量评估指标
        
        Args:
            predictions: [N] 预测分数
            targets: [N] 真实分数
        """
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # 相关性
        if len(predictions) > 1:
            corr, _ = pearsonr(predictions.flatten(), targets.flatten())
        else:
            corr = 0.0
        
        # 准确率（按阈值）
        threshold = 5.0
        accuracy = np.mean(np.abs(predictions - targets) < threshold)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(corr),
            'accuracy': float(accuracy)
        }
    
    @staticmethod
    def compute_error_detection_metrics(pred_errors: np.ndarray,
                                        true_errors: np.ndarray) -> Dict[str, float]:
        """
        计算错误检测指标（多标签分类）
        
        Args:
            pred_errors: [N, num_errors] 预测概率
            true_errors: [N, num_errors] 真实标签
        """
        # 二值化预测
        pred_binary = (pred_errors > 0.5).astype(int)
        
        # 样本级准确率
        sample_acc = np.mean(np.all(pred_binary == true_errors, axis=1))
        
        # 标签级指标
        tp = np.sum((pred_binary == 1) & (true_errors == 1))
        fp = np.sum((pred_binary == 1) & (true_errors == 0))
        fn = np.sum((pred_binary == 0) & (true_errors == 1))
        tn = np.sum((pred_binary == 0) & (true_errors == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC（简化计算）
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(true_errors.flatten(), pred_errors.flatten())
        except:
            auc = 0.5
        
        return {
            'sample_accuracy': float(sample_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc)
        }


def levenshtein_distance(seq1: np.ndarray, seq2: np.ndarray) -> int:
    """
    计算编辑距离
    """
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # 删除
                dp[i, j - 1] + 1,      # 插入
                dp[i - 1, j - 1] + cost  # 替换
            )
    
    return dp[m, n]


def compute_scores(metrics: Dict, action_config: Dict) -> Tuple[float, Dict]:
    """
    根据指标计算得分
    
    Args:
        metrics: 各项指标值
        action_config: 动作配置
    
    Returns:
        overall_score: 总分
        details: 详细得分
    """
    standards = action_config.get('standard_params', {})
    scores = {}
    
    for param_name, standard in standards.items():
        if param_name not in metrics:
            continue
        
        value = metrics[param_name]
        min_val = standard.get('min', 0)
        max_val = standard.get('max', 100)
        ideal = standard.get('ideal', (min_val + max_val) / 2)
        weight = standard.get('weight', 0.1)
        
        # 计算偏离理想值的程度
        if min_val <= value <= max_val:
            # 在范围内
            range_size = max_val - min_val
            if range_size > 0:
                deviation = abs(value - ideal) / range_size
                score = max(0, 100 - deviation * 50)
            else:
                score = 100.0
        else:
            # 超出范围
            deviation = abs(value - ideal) / (max_val - min_val + 1e-6)
            score = max(0, 50 - deviation * 25)
        
        scores[param_name] = {
            'value': float(value),
            'score': float(score),
            'weight': float(weight)
        }
    
    # 加权计算总分
    if scores:
        total_weight = sum(s['weight'] for s in scores.values())
        if total_weight > 0:
            overall = sum(s['score'] * s['weight'] for s in scores.values()) / total_weight
        else:
            overall = 50.0
    else:
        overall = 50.0
    
    # 根据错误扣分
    errors = metrics.get('errors', [])
    error_count = len(errors)
    
    # 根据错误严重程度扣分
    penalty = 0
    for error in errors:
        # 简单处理：每个错误扣5分
        penalty += 5
    
    overall = max(0, overall - min(penalty, 30))  # 最多扣30分
    
    return overall, scores


def get_grade(score: float) -> str:
    """根据分数评定等级"""
    if score >= 90:
        return '优秀'
    elif score >= 80:
        return '良好'
    elif score >= 60:
        return '及格'
    else:
        return '不及格'


def format_assessment_report(action_type: str,
                             overall_score: float,
                             details: Dict,
                             errors: List[str],
                             phases: List[int] = None) -> str:
    """
    格式化评估报告
    """
    report = []
    report.append("=" * 50)
    report.append(f"动作评估报告: {action_type}")
    report.append("=" * 50)
    report.append(f"\n总体得分: {overall_score:.1f} 分")
    report.append(f"等级评定: {get_grade(overall_score)}")
    
    if details:
        report.append("\n各维度得分:")
        for param_name, info in details.items():
            report.append(f"  {param_name}: {info['score']:.1f} 分 "
                         f"(值: {info['value']:.2f}, 权重: {info['weight']:.2f})")
    
    if errors:
        report.append(f"\n检测到 {len(errors)} 个错误:")
        for error in errors:
            report.append(f"  - {error}")
    else:
        report.append("\n未检测到明显错误")
    
    if phases is not None:
        report.append(f"\n动作阶段数: {len(set(phases))}")
    
    report.append("=" * 50)
    
    return '\n'.join(report)


class MetricsTracker:
    """训练指标追踪器"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_best(self, metric: str, mode: str = 'max') -> Tuple[int, float]:
        """获取最佳指标值"""
        values = self.history.get(metric, [])
        if not values:
            return -1, 0.0
        
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return best_idx, values[best_idx]
    
    def save(self, path: str):
        """保存历史记录"""
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot(self, save_path: str = None):
        """绘制训练曲线"""
        import matplotlib.pyplot as plt
        
        num_metrics = len(self.history)
        fig, axes = plt.subplots(
            (num_metrics + 1) // 2, 2, 
            figsize=(12, 3 * ((num_metrics + 1) // 2))
        )
        axes = axes.flatten() if num_metrics > 1 else [axes]
        
        for idx, (key, values) in enumerate(self.history.items()):
            if values:
                axes[idx].plot(values)
                axes[idx].set_title(key)
                axes[idx].set_xlabel('Epoch')
                axes[idx].grid(True)
        
        # 隐藏多余的子图
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
