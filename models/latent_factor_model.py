import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

class LatentFactorModel:
    def __init__(self, num_factors=32):
        """
        初始化隐语义模型。
        
        参数：
        - num_factors: 隐因子的数量（即嵌入维度）
        """
        self.num_factors = num_factors
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, data):
        """
        训练模型，计算用户和物品的隐因子。
        
        参数：
        - data: 包含 user_id, movie_id 和 rating 的 DataFrame
        """
        # 构建评分矩阵
        rating_matrix = data.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
        
        # 使用 TruncatedSVD 进行矩阵分解
        svd = TruncatedSVD(n_components=self.num_factors, random_state=42)
        self.user_factors = svd.fit_transform(rating_matrix)
        self.item_factors = svd.components_.T
        
        # 将隐因子存储为 DataFrame
        self.user_factors = pd.DataFrame(self.user_factors, index=rating_matrix.index)
        self.item_factors = pd.DataFrame(self.item_factors, index=rating_matrix.columns)
    
    def recommend(self, user_id, top_k=10):
        """
        为指定用户生成推荐列表。
        
        参数：
        - user_id: 用户 ID
        - top_k: 推荐数量
        
        返回：
        - 推荐的电影 ID 列表
        """
        if user_id not in self.user_factors.index:
            raise ValueError(f"User ID {user_id} not found in the user factors.")
        
        # 获取用户的隐因子向量
        user_vector = self.user_factors.loc[user_id].values.reshape(1, -1)
        
        # 计算用户对所有物品的预测评分
        predictions = np.dot(user_vector, self.item_factors.T).flatten()
        recommendations = pd.Series(predictions, index=self.item_factors.index)
        
        # 排序并返回 Top-K 推荐
        top_movies = recommendations.sort_values(ascending=False).index[:top_k].tolist()
        return top_movies