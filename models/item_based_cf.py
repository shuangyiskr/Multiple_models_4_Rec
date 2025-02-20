import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedCF:
    def __init__(self, similarity_metric="cosine"):
        """
        初始化基于物品的协同过滤模型。
        
        参数：
        - similarity_metric: 相似度计算方法，默认为余弦相似度（"cosine"）。
        """
        self.similarity_metric = similarity_metric
        self.item_similarity_matrix = None
        self.rating_matrix = None
    
    def fit(self, data):
        """
        训练模型，计算物品相似度矩阵。
        
        参数：
        - data: 包含 user_id, movie_id 和 rating 的 DataFrame
        """
        # 构建评分矩阵
        self.rating_matrix = data.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
        
        # 计算物品相似度矩阵
        if self.similarity_metric == "cosine":
            self.item_similarity_matrix = cosine_similarity(self.rating_matrix.T)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.rating_matrix.columns,
            columns=self.rating_matrix.columns
        )
    
    def recommend(self, user_id, top_k=10):
        """
        为指定用户生成推荐列表。
        
        参数：
        - user_id: 用户 ID
        - top_k: 推荐数量
        
        返回：
        - 推荐的电影 ID 列表
        """
        if user_id not in self.rating_matrix.index:
            raise ValueError(f"User ID {user_id} not found in the rating matrix.")
        
        # 获取当前用户的评分向量
        user_ratings = self.rating_matrix.loc[user_id]
        
        # 计算加权评分
        weighted_scores = user_ratings.values.reshape(-1, 1) * self.item_similarity_matrix.values
        weighted_scores = pd.DataFrame(weighted_scores, index=self.rating_matrix.columns, columns=self.rating_matrix.columns)
        
        # 过滤掉用户已经评分过的电影
        recommendations = weighted_scores.sum(axis=1) / (user_ratings.sum() + 1e-8)
        recommendations = recommendations[user_ratings == 0]  # 只推荐未评分的电影
        
        # 排序并返回 Top-K 推荐
        top_movies = recommendations.sort_values(ascending=False).index[:top_k].tolist()
        return top_movies