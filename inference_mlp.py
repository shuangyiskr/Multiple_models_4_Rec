import os
import torch
import pandas as pd
from tqdm import tqdm
from models.mlp_model import MLPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# 定义数据路径
PROCESSED_DATA_PATH = "data/processed_data.pkl"
# MODEL_PATH = "models/mlp_model.pth"

MODEL_PATH = "models/last_checkpoint.pth"
# 定义超参数
BATCH_SIZE = 64
TOP_K = 10  # 每个用户的推荐数量

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
    # 加载数据
    test_data, num_users, num_movies = load_processed_data()
    
    # 初始化模型并加载权重
    model = MLPModel(num_users=num_users, num_movies=num_movies).to(device)

    checkpoint = torch.load(MODEL_PATH)
    # 打印检查点内容
    print("Keys in checkpoint:", checkpoint.keys())
    for key, value in checkpoint.items():
        print(f"{key}: {type(value)}")
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # 设置为评估模式
    
    # 获取所有用户和电影 ID
    all_user_ids = torch.arange(num_users, dtype=torch.long).to(device)  # 所有用户 ID
    all_movie_ids = torch.arange(num_movies, dtype=torch.long).to(device)  # 所有电影 ID
    
    # 推理过程
    recommendations = {}
    progress_bar = tqdm(all_user_ids, desc="Generating Recommendations")
    for user_id in progress_bar:
        # 全量召回：对当前用户的所有电影进行评分预测
        user_ids = torch.full_like(all_movie_ids, user_id)  # 用户 ID 扩展为与电影 ID 相同长度
        with torch.no_grad():
            predictions = model(user_ids, all_movie_ids).cpu().numpy()
        
        # 排序并获取 Top-K 推荐
        top_k_indices = predictions.argsort()[-TOP_K:][::-1]  # 取评分最高的 K 个电影
        recommendations[user_id.item()] = top_k_indices.tolist()
    
    # 保存推荐结果
    os.makedirs("results", exist_ok=True)
    with open("results/recommendations_emb16_4096.txt", "w") as f:
        for user_id, movie_ids in recommendations.items():
            f.write(f"User {user_id}: {movie_ids}\n")
    
    print("Recommendations saved to results/recommendations_emb16_4096.txt")

if __name__ == "__main__":
    inference()