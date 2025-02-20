import torch
import torch.nn as nn

class YouTubeDNN(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32, hidden_dims=[64, 32], dropout_rate=0.2):
        """
        初始化 YouTubeDNN 模型。
        
        参数：
        - num_users: 用户数量
        - num_movies: 电影数量
        - embedding_dim: 嵌入维度
        - hidden_dims: 隐藏层维度列表
        - dropout_rate: Dropout 比率
        """
        super(YouTubeDNN, self).__init__()
        
        # 用户嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # MLP 层
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # 输出层（预测电影）
        layers.append(nn.Linear(hidden_dims[-1], num_movies))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, user_ids):
        """
        前向传播。
        
        参数：
        - user_ids: 用户 ID 张量 (batch_size,)
        
        返回：
        - logits: 对所有电影的预测分数 (batch_size, num_movies)
        """
        user_embeds = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        logits = self.mlp(user_embeds)  # (batch_size, num_movies)
        return logits