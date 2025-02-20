import os
import pandas as pd
from tqdm import tqdm
from models.user_based_cf import UserBasedCF

# 定义数据路径
PROCESSED_DATA_PATH = "data/processed_data.pkl"
MODEL_PATH = "models/ubcf_model.pkl"
RECOMMENDATIONS_PATH = "results/ubcf_recommendations.txt"

def load_processed_data():
    """
    加载预处理后的数据。
    """
    processed_data = pd.read_pickle(PROCESSED_DATA_PATH)
    test_data = processed_data["test_data"]
    return test_data

def inference():
    # 加载测试数据
    test_data = load_processed_data()
    
    # 加载模型
    import pickle
    with open(MODEL_PATH, "rb") as f:
        ubcf_model = pickle.load(f)
    
    # 生成推荐结果
    print("Generating recommendations...")
    recommendations = {}
    for user_id in tqdm(test_data["user_id"].unique(), desc="Generating Recommendations"):
        try:
            recommendations[user_id] = ubcf_model.recommend(user_id, top_k=10)
        except ValueError:
            recommendations[user_id] = []
    print("Recommendations generated.")
    
    # 保存推荐结果
    os.makedirs("results", exist_ok=True)
    with open(RECOMMENDATIONS_PATH, "w") as f:
        for user_id, movie_ids in recommendations.items():
            f.write(f"User {user_id}: {movie_ids}\n")
    print(f"Recommendations saved to {RECOMMENDATIONS_PATH}")

if __name__ == "__main__":
    inference()