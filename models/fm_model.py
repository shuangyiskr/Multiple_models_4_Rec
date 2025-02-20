import torch
import torch.nn as nn

class FMModel(nn.Module):
    def __init__(self, num_features, embedding_dim=8):
        """
        初始化 FM 模型。
        
        参数：
        - num_features: 特征数量（用户数 + 电影数）
        - embedding_dim: 嵌入维度
        """
        super(FMModel, self).__init__()
        
        # 线性部分
        self.linear = nn.Embedding(num_features, 1)
        
        # 二阶交互部分
        self.embedding = nn.Embedding(num_features, embedding_dim)
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, feature_indices):
        """
        前向传播。
        
        参数：
        - feature_indices: 特征索引（用户 ID 和电影 ID 的拼接），形状为 (batch_size, num_features)
        
        返回：
        - 预测评分
        """
        # 线性部分
        linear_part = self.linear(feature_indices).sum(dim=1).squeeze()
        
        # 二阶交互部分
        embeddings = self.embedding(feature_indices)  # (batch_size, num_features, embedding_dim)
        square_of_sum = embeddings.sum(dim=1).pow(2)  # (batch_size, embedding_dim)
        sum_of_square = (embeddings.pow(2)).sum(dim=1)  # (batch_size, embedding_dim)
        interaction_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)  # (batch_size,)
        
        # 总输出
        output = linear_part + interaction_part + self.bias
        return output