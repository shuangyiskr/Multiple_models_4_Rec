import torch
import torch.nn as nn

class EGES(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim, num_side_info):
        super(EGES, self).__init__()
        
        # 主节点嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # 侧信息嵌入
        self.side_info_embedding = nn.Embedding(num_side_info, embedding_dim)
        
        # 权重参数
        self.alpha = nn.Parameter(torch.ones(embedding_dim))
    
    def forward(self, user_ids, movie_ids, side_info_ids):
        """
        前向传播：
        - user_ids: 用户 ID
        - movie_ids: 电影 ID
        - side_info_ids: 侧信息 ID
        """
        # 获取主节点嵌入
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        
        # 获取侧信息嵌入
        side_info_embed = self.side_info_embedding(side_info_ids)
        
        # 聚合主节点嵌入和侧信息嵌入
        combined_embed = torch.mul(user_embed + movie_embed, self.alpha) + side_info_embed
        
        # 计算相似度（内积）
        similarity = torch.sum(user_embed * movie_embed, dim=1)
        
        return similarity