import torch
import torch.nn as nn
import torch.nn.init as init

class MLPModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32, hidden_dims=[64, 32], dropout_rate=0.2):
    # def __init__(self, num_users, num_movies, embedding_dim=16, hidden_dims=[32], dropout_rate=0.2):
        """
        初始化 MLP 模型。
        
        参数：
        - num_users: 用户数量
        - num_movies: 电影数量
        - embedding_dim: 嵌入向量的维度
        - hidden_dims: MLP 隐藏层的维度列表
        - dropout_rate: Dropout 概率
        """
        super(MLPModel, self).__init__()
        
        # 用户和电影的嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # 使用 Xavier 初始化
        init.xavier_uniform_(self.user_embedding.weight)
        init.xavier_uniform_(self.movie_embedding.weight)
        
        # MLP 层
        layers = []
        input_dim = 2 * embedding_dim  # 用户嵌入 + 电影嵌入
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            init.xavier_uniform_(layers[-1].weight)  # 初始化线性层权重
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], 1))  # 输出一个标量值（预测评分）
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, user_ids, movie_ids):
        """
        前向传播函数。
        
        参数：
        - user_ids: 用户 ID 张量 (batch_size,)
        - movie_ids: 电影 ID 张量 (batch_size,)
        
        返回：
        - 预测评分张量 (batch_size, 1)
        """
        # 获取用户和电影的嵌入向量
        user_embeds = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        movie_embeds = self.movie_embedding(movie_ids)  # (batch_size, embedding_dim)
        
        # 拼接用户和电影的嵌入向量
        concat_embeds = torch.cat([user_embeds, movie_embeds], dim=1)  # (batch_size, 2 * embedding_dim)
        
        # 通过 MLP 层得到预测评分
        predictions = self.mlp(concat_embeds)  # (batch_size, 1)

        # 裁剪预测值到 [1, 5] 范围
        predictions = torch.clamp(predictions.squeeze(), min=0, max=1)

        # 评估指标
        # print("User Embeddings:", user_embeds)
        # print("Movie Embeddings:", movie_embeds)
        # print("Concatenated Embeddings:", concat_embeds)
        # print("MLP Output:", predictions)
    
        return predictions.squeeze()  # 返回形状为 (batch_size,)