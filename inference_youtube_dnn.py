import os
import torch
import pandas as pd
from tqdm import tqdm
from models.youtube_dnn import YouTubeDNN

# 定义数据路径
PROCESSED_DATA_PATH = "data/processed_data.pkl"
MODEL_PATH = "models/youtube_dnn.pth"
RECOMMENDATIONS_PATH = "results/youtube_dnn_recommendations.txt"

# 定义超参数
TOP_K = 10

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
    model = YouTubeDNN(num_users=num_users, num_movies=num_movies).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 生成推荐结果
    recommendations = {}
    with torch.no_grad():
        for user_id in tqdm(range(num_users), desc="Generating Recommendations"):
            user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
            logits = model(user_tensor)
            top_k_scores, top_k_movies = torch.topk(logits, TOP_K)
            recommendations[user_id] = top_k_movies.cpu().numpy().tolist()
    
    # 保存推荐结果
    os.makedirs("results", exist_ok=True)
    with open(RECOMMENDATIONS_PATH, "w") as f:
        for user_id, movie_ids in recommendations.items():
            f.write(f"User {user_id}: {movie_ids[0]}\n")
    print(f"Recommendations saved to {RECOMMENDATIONS_PATH}")

if __name__ == "__main__":
    inference()