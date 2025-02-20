import os
import torch
import pandas as pd
from tqdm import tqdm
from models.fm_model import FMModel

# 定义数据路径
PROCESSED_DATA_PATH = "data/processed_data.pkl"
MODEL_PATH = "models/fm_model.pth"
RECOMMENDATIONS_PATH = "results/fm_recommendations.txt"

# 定义超参数
TOP_K = 10
EMBEDDING_DIM = 8

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_processed_data():
    """
    加载预处理后的数据。
    """
    processed_data = pd.read_pickle(PROCESSED_DATA_PATH)
    test_data = processed_data["test_data"]
    user_id_map = processed_data["user_id_map"]
    movie_id_map = processed_data["movie_id_map"]
    
    return test_data, len(user_id_map), len(movie_id_map)

def inference():
    # 加载测试数据
    test_data, num_users, num_movies = load_processed_data()
    
    # 初始化模型并加载权重
    model = FMModel(num_features=num_users + num_movies, embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 生成推荐结果
    recommendations = {}
    all_movie_ids = torch.arange(num_movies, dtype=torch.long) + num_users  # 将电影 ID 映射到用户之后的范围
    with torch.no_grad():
        for user_id in tqdm(range(num_users), desc="Generating Recommendations"):
            user_ids = torch.full_like(all_movie_ids, user_id, dtype=torch.long)
            feature_indices = torch.stack([user_ids, all_movie_ids], dim=1).to(device)
            predictions = model(feature_indices).cpu().numpy()
            
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