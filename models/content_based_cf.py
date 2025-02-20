import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedCF:
    def __init__(self):
        """
        初始化基于内容的协同过滤模型。
        """
        self.user_profile = None
        self.movie_content_features = None
    
    def fit(self, train_data, movie_content_features):
        """
        训练模型，构建用户画像。
        
        参数：
        - train_data: 包含 user_id, movie_id 和 rating 的 DataFrame
        - movie_content_features: 电影内容特征矩阵
        """
        self.movie_content_features = movie_content_features
        
        # 构建用户画像
        user_profiles = []
        for user_id in train_data["user_id"].unique():
            user_ratings = train_data[train_data["user_id"] == user_id]
            weighted_features = (
                user_ratings["rating"].values.reshape(-1, 1) * 
                self.movie_content_features.loc[user_ratings["movie_id"].values].values
            )
            user_profile = weighted_features.sum(axis=0) / (user_ratings["rating"].sum() + 1e-8)
            user_profiles.append(user_profile)
        
        self.user_profile = pd.DataFrame(user_profiles, index=train_data["user_id"].unique(), columns=movie_content_features.columns)
    
    def recommend(self, user_id, top_k=10):
        """
        为指定用户生成推荐列表。
        
        参数：
        - user_id: 用户 ID
        - top_k: 推荐数量
        
        返回：
        - 推荐的电影 ID 列表
        """
        if user_id not in self.user_profile.index:
            raise ValueError(f"User ID {user_id} not found in the user profile.")
        
        # 计算用户与所有电影的相似度
        user_vector = self.user_profile.loc[user_id].values.reshape(1, -1)
        similarities = cosine_similarity(user_vector, self.movie_content_features.values)[0]
        
        # 排序并返回 Top-K 推荐
        top_movie_ids = np.argsort(similarities)[::-1][:top_k]
        return top_movie_ids.tolist()