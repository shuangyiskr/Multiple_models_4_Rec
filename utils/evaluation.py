import numpy as np
from collections import defaultdict

def calculate_metrics(test_data, recommendations, top_k=10):
    """
    计算 Precision@K, Recall@K 和 NDCG@K。
    
    参数：
    - test_data: 测试集数据，包含 user_id, movie_id 和 rating 的 DataFrame
    - recommendations: 每个用户的推荐结果，字典格式 {user_id: [movie_ids]}
    - top_k: 考虑的推荐数量
    
    返回：
    - precision, recall, ndcg: 平均值
    """
    # 构建用户的真实交互集合
    user_ground_truth = defaultdict(set)
    for _, row in test_data.iterrows():
        user_ground_truth[row["user_id"]].add(row["movie_id"])
    
    # 初始化指标
    precision_list = []
    recall_list = []
    ndcg_list = []
    
    for user_id, recommended_items in recommendations.items():
        ground_truth = user_ground_truth[user_id]
        if not ground_truth:  # 如果用户没有测试集中的交互，跳过
            continue
        
        # 截取 Top-K 推荐
        recommended_items = recommended_items[:top_k]
        
        # 计算命中数
        hits = len(set(recommended_items) & ground_truth)
        
        # Precision@K
        precision = hits / top_k
        precision_list.append(precision)
        
        # Recall@K
        recall = hits / len(ground_truth)
        recall_list.append(recall)
        
        # NDCG@K
        dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(recommended_items) if item in ground_truth])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(ground_truth), top_k))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)
    
    # 返回平均值
    return (
        np.mean(precision_list),
        np.mean(recall_list),
        np.mean(ndcg_list)
    )