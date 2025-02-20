import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
from models.eges import EGES
from torch.utils.data import DataLoader, Dataset
from utils.graph_builder import build_graph
from models.eges import EGES
import os

# 超参数
EMBEDDING_DIM = 64
NUM_SIDE_INFO = 100  # 示例值，根据实际侧信息数量调整
RECOMMENDATIONS_PATH = "results/eges_recommendations.txt"
TOP_K =10

def inference():
    # 加载预处理后的数据
    processed_data = pd.read_pickle("data/processed_data.pkl")
    test_data = processed_data["test_data"]
    user_id_map = processed_data["user_id_map"]
    movie_id_map = processed_data["movie_id_map"]
    
    # 构建图
    users = pd.read_csv("data/ml-1m/users.dat", sep="::", engine="python", header=None, names=["user_id", "gender", "age", "occupation", "zip_code"])
    movies = pd.read_csv("data/ml-1m/movies.dat", sep="::", engine="python", header=None, names=["movie_id", "title", "genres"], encoding="ISO-8859-1")
    graph = build_graph(test_data, users, movies)
    
    # 初始化模型
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)
    model = EGES(num_users, num_movies, EMBEDDING_DIM, NUM_SIDE_INFO)
    
    # 加载训练好的模型
    model.load_state_dict(torch.load("models/eges.pth"))
    model.eval()
    
    # 为每个用户生成推荐列表
    recommendations = {}
    for user_id in tqdm(user_id_map.values()):
        user_ids = torch.tensor([user_id] * num_movies, dtype=torch.long)
        movie_ids = torch.arange(num_movies, dtype=torch.long)
        side_info_ids = torch.randint(0, NUM_SIDE_INFO, (num_movies,), dtype=torch.long)
        
        with torch.no_grad():
            predictions = model(user_ids, movie_ids, side_info_ids).cpu().numpy()
            # 排序并获取 Top-K 推荐
            top_k_indices = predictions.argsort()[-TOP_K:][::-1]
            recommendations[user_id] = top_k_indices.tolist()
    
    # 保存推荐结果
    os.makedirs("results", exist_ok=True)
    with open(RECOMMENDATIONS_PATH, "w") as f:
        for user_id, movie_ids in recommendations.items():
            f.write(f"User {user_id}: {movie_ids}\n")
    print(f"Recommendations saved to {RECOMMENDATIONS_PATH}")

if __name__ == "__main__":
    inference()