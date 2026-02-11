# @author Coder建设｜javpower
"""
深度学习模型定义
包含：动作识别、阶段分割、质量评估三个模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


# ==================== 动作识别模型 ====================

class STGCNAction(nn.Module):
    """
    基于时空图卷积网络的动作识别模型
    输入：骨骼序列 [B, T, N, C]
    输出：动作类别概率 [B, num_classes]
    """
    
    def __init__(self, num_classes: int = 6, in_channels: int = 9, 
                 edge_importance_weighting: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 人体骨骼邻接矩阵
        self.register_buffer('A', self._build_adjacency())
        
        # 构建ST-GCN层
        self.st_gcn_layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, kernel_size=3, stride=1),
            STGCNBlock(64, 64, kernel_size=3, stride=1),
            STGCNBlock(64, 128, kernel_size=3, stride=2),
            STGCNBlock(128, 128, kernel_size=3, stride=1),
            STGCNBlock(128, 256, kernel_size=3, stride=2),
            STGCNBlock(256, 256, kernel_size=3, stride=1),
        ])
        
        # 边重要性权重（可学习）
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_layers
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_layers)
        
        # 全局池化和分类
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 初始化
        self._init_weights()
    
    def _build_adjacency(self) -> torch.Tensor:
        """构建归一化的邻接矩阵"""
        # 人体骨骼连接定义
        skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [5, 11], [6, 12], [11, 12],
            [11, 13], [13, 15], [12, 14], [14, 16]
        ]
        
        A = np.zeros((17, 17))
        for i, j in skeleton:
            A[i, j] = A[j, i] = 1
        
        # 归一化
        D = np.sum(A, axis=1) + 1e-6
        D_sqrt_inv = np.diag(np.power(D, -0.5))
        A_norm = D_sqrt_inv @ A @ D_sqrt_inv
        
        return torch.FloatTensor(A_norm)
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - 批次、时间、关节、通道
        Returns:
            [B, num_classes] - 动作类别 logits
        """
        # 调整维度: [B, T, N, C] -> [B, C, T, N]
        x = x.permute(0, 3, 1, 2)
        
        # ST-GCN层
        for layer, importance in zip(self.st_gcn_layers, self.edge_importance):
            x = layer(x, self.A * importance)
        
        # 全局池化
        x = self.global_pool(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        
        # 分类
        x = self.fc(x)
        
        return x


class STGCNBlock(nn.Module):
    """时空图卷积块"""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 stride: int = 1, dropout: float = 0.3):
        super().__init__()
        
        # 确保kernel_size是奇数
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = (kernel_size - 1) // 2
        
        # 图卷积（1x1卷积实现）
        self.graph_conv = nn.Conv2d(in_ch, out_ch, 1)
        
        # 时序卷积
        self.temporal_conv = nn.Conv2d(
            out_ch, out_ch, (kernel_size, 1),
            stride=(stride, 1), padding=(padding, 0)
        )
        
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.residual = lambda x: x
    
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, N]
            A: [N, N] 邻接矩阵
        """
        B, C, T, N = x.shape
        
        # 图卷积: [B, C, T, N] -> [B, C, T*N]
        x_flat = x.view(B, C, T * N)
        
        # 图卷积操作: X' = A @ X
        x_graph = torch.einsum('ncv,vw->ncw', x_flat, A)
        x_graph = x_graph.view(B, C, T, N)
        x_graph = self.graph_conv(x_graph)
        
        # 时序卷积
        out = self.temporal_conv(x_graph)
        out = self.bn(out)
        
        # 残差连接
        res = self.residual(x)
        out = self.relu(out + res)
        out = self.dropout(out)
        
        return out


# ==================== 阶段分割模型 ====================

class TemporalUNet(nn.Module):
    """
    基于时序U-Net的阶段分割模型
    输入：骨骼序列 [B, T, N, C]
    输出：阶段标签 [B, T, num_phases]
    """
    
    def __init__(self, in_channels: int = 153, num_phases: int = 5,
                 hidden_dims: list = [64, 128, 256, 512]):
        super().__init__()
        
        self.num_phases = num_phases
        
        # 输入投影
        self.input_proj = nn.Conv1d(in_channels, hidden_dims[0], 1)
        
        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.encoders.append(
                self._conv_block(hidden_dims[i], hidden_dims[i+1])
            )
            self.pools.append(nn.MaxPool1d(2))
        
        # 瓶颈
        self.bottleneck = self._conv_block(hidden_dims[-1], hidden_dims[-1] * 2)
        
        # 解码器
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose1d(
                    hidden_dims[i] * 2 if i == len(hidden_dims) - 1 else hidden_dims[i],
                    hidden_dims[i-1], 2, stride=2
                )
            )
            self.decoders.append(
                self._conv_block(hidden_dims[i], hidden_dims[i-1])
            )
        
        # 输出层
        self.out = nn.Conv1d(hidden_dims[0], num_phases, 1)
        
        # 时序注意力
        self.temporal_attention = TemporalAttention(hidden_dims[0])
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """卷积块"""
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - 批次、时间、关节、通道
        Returns:
            [B, T, num_phases] - 每帧的阶段 logits
        """
        B, T, N, C = x.shape
        
        # 展平关节和特征: [B, T, N*C] -> [B, N*C, T]
        x = x.reshape(B, T, N * C).permute(0, 2, 1)
        
        # 输入投影
        x = self.input_proj(x)
        
        # 编码
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # 瓶颈
        x = self.bottleneck(x)
        
        # 解码
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skip_connections)):
            x = upconv(x)
            # 处理尺寸不匹配
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, [diff // 2, diff - diff // 2])
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # 时序注意力
        x = self.temporal_attention(x)
        
        # 输出
        x = self.out(x).permute(0, 2, 1)  # [B, T, num_phases]
        
        return x


class TemporalAttention(nn.Module):
    """时序注意力模块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        """
        B, C, T = x.shape
        
        # 计算注意力
        q = self.query(x).permute(0, 2, 1)  # [B, T, C//8]
        k = self.key(x)  # [B, C//8, T]
        v = self.value(x)  # [B, C, T]
        
        attention = torch.bmm(q, k)  # [B, T, T]
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, T]
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


# ==================== 质量评估模型 ====================

class QualityNet(nn.Module):
    """
    多任务质量评估模型
    输入：骨骼序列 [B, T, N, C]
    输出：
        - 整体评分 [B, 1]
        - 各维度评分 [B, 4]
        - 错误检测 [B, num_errors]
    """
    
    def __init__(self, num_metrics: int = 4, num_errors: int = 27,
                 feature_dim: int = 256, input_frames: int = 60,
                 input_joints: int = 17, input_channels: int = 9):
        super().__init__()
        
        self.num_metrics = num_metrics
        self.num_errors = num_errors
        
        # 时空特征提取器
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_joints * input_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        # 时序编码器（LSTM）
        self.temporal_encoder = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # 注意力池化
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 共享特征
        self.feature_fusion = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 多任务头
        # 1. 整体评分 (0-100)
        self.score_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出0-1，乘以100
        )
        
        # 2. 各维度评分 (准确性、稳定性、标准性、安全性)
        self.metric_heads = nn.ModuleDict({
            'accuracy': nn.Linear(feature_dim, 1),
            'stability': nn.Linear(feature_dim, 1),
            'standard': nn.Linear(feature_dim, 1),
            'safety': nn.Linear(feature_dim, 1),
        })
        
        # 3. 错误检测（多标签分类）
        self.error_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_errors),
            nn.Sigmoid()
        )
        
        # 4. 是否标准动作（二分类）
        self.standard_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, N, C]
        Returns:
            dict: 包含各项预测结果
        """
        B, T, N, C = x.shape
        
        # 空间编码: [B, T, N, C] -> [B, T, N*C] -> [B, T, 128]
        x_flat = x.reshape(B, T, N * C)
        spatial_features = self.spatial_encoder(x_flat)
        
        # 时序编码: [B, T, 128] -> [B, T, 256] (双向LSTM)
        temporal_features, _ = self.temporal_encoder(spatial_features)
        
        # 注意力池化
        attn_weights = F.softmax(self.attention(temporal_features), dim=1)  # [B, T, 1]
        context = torch.sum(temporal_features * attn_weights, dim=1)  # [B, 256]
        
        # 特征融合
        features = self.feature_fusion(context)
        
        # 多任务预测
        overall_score = self.score_head(features) * 100
        
        metric_scores = {
            name: torch.sigmoid(head(features)) * 100
            for name, head in self.metric_heads.items()
        }
        
        error_probs = self.error_head(features)
        is_standard = self.standard_head(features)
        
        return {
            'overall': overall_score,
            'metrics': metric_scores,
            'errors': error_probs,
            'is_standard': is_standard,
            'attention_weights': attn_weights
        }


# ==================== 轻量级模型（移动端部署） ====================

class LiteActionNet(nn.Module):
    """轻量级动作识别模型（用于边缘设备）"""
    
    def __init__(self, num_classes: int = 6):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(60 * 17 * 9, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.reshape(B, -1)
        x = self.features(x)
        x = self.classifier(x)
        return x


class LiteQualityNet(nn.Module):
    """轻量级质量评估模型"""
    
    def __init__(self, num_errors: int = 27):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(60 * 17 * 9, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.error_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_errors),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        x = x.reshape(B, -1)
        features = self.encoder(x)
        
        return {
            'overall': self.score_head(features) * 100,
            'errors': self.error_head(features)
        }


# ==================== 模型工具函数 ====================

def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: tuple = (1, 60, 17, 9)):
    """打印模型摘要"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # 测试前向传播
    x = torch.randn(*input_size)
    with torch.no_grad():
        y = model(x)
    
    if isinstance(y, dict):
        print("Output shapes:")
        for k, v in y.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2.shape}")
    else:
        print(f"Output shape: {y.shape}")


def export_to_onnx(model: nn.Module, save_path: str, 
                   input_size: tuple = (1, 60, 17, 9)):
    """导出模型为ONNX格式"""
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"Model exported to {save_path}")


if __name__ == '__main__':
    # 测试模型
    print("=" * 50)
    print("Testing STGCNAction")
    model1 = STGCNAction(num_classes=6)
    model_summary(model1)
    
    print("\n" + "=" * 50)
    print("Testing TemporalUNet")
    model2 = TemporalUNet(num_phases=5)
    model_summary(model2)
    
    print("\n" + "=" * 50)
    print("Testing QualityNet")
    model3 = QualityNet(num_errors=27)
    model_summary(model3)
