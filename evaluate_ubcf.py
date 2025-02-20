import os
import pandas as pd
from utils.evaluation import calculate_metrics

# 定义数据路径
PROCESSED_DATA_PATH = "data/processed_data.pkl"
RECOMMENDATIONS_PATH = "results/ubcf_recommendations.txt"

def load_processed_data():
    """
    加载预处理后的数据。
    """
    processed_data = pd.read_pickle(PROCESSED_DATA_PATH)
    test_data = processed_data["test_data"]
    return test_data

def load_recommendations():
    """
    加载推荐结果。
    """
    recommendations = {}
    with open(RECOMMENDATIONS_PATH, "r") as f:
        for line in f:
            parts = line.strip().split(": ")
            user_id = int(parts[0].split(" ")[1])
            movie_ids = list(map(int, parts[1][1:-1].split(", ")))
            recommendations[user_id] = movie_ids
    return recommendations

def evaluate():
    # 加载测试数据和推荐结果
    test_data = load_processed_data()
    recommendations = load_recommendations()
    
    # 计算评价指标
    precision, recall, ndcg = calculate_metrics(test_data, recommendations, top_k=10)
    
    # 输出结果
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")

if __name__ == "__main__":
    evaluate()