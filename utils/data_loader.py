import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 定义数据路径
DATA_DIR = "data/ml-1m"
PROCESSED_DATA_PATH = "data/processed_data.pkl"

def load_data():
    """
    加载 Movielens-1m 数据集中的 users, movies 和 ratings 数据。
    """
    # 加载用户数据
    users = pd.read_csv(
        os.path.join(DATA_DIR, "users.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"]
    )
    
    # 加载电影数据
    movies = pd.read_csv(
        os.path.join(DATA_DIR, "movies.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="ISO-8859-1"  # 解决编码问题
    )
    
    # 加载评分数据
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "ratings.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    
    return users, movies, ratings

def preprocess_data(users, movies, ratings):
    """
    对数据进行预处理：
    1. 将 user_id 和 movie_id 转换为连续索引。
    2. 划分训练集和测试集。
    """
    print("nan总数：")
    print(ratings.isnull().sum())
    # 删除包含 NaN 值的行
    ratings = ratings.dropna()
    
    # 将 user_id 和 movie_id 转换为连续索引
    user_id_map = {id: idx for idx, id in enumerate(users["user_id"].unique())}
    movie_id_map = {id: idx for idx, id in enumerate(movies["movie_id"].unique())}
    
    ratings["user_id"] = ratings["user_id"].map(user_id_map)
    ratings["movie_id"] = ratings["movie_id"].map(movie_id_map)
    
    # 尝试归一化平均分
    ratings["rating"] = (ratings["rating"] - ratings["rating"].min()) / (ratings["rating"].max() - ratings["rating"].min())
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    
    return train_data, test_data, user_id_map, movie_id_map

def save_processed_data(train_data, test_data, user_id_map, movie_id_map):
    """
    保存预处理后的数据到 processed_data.pkl 文件。
    """
    processed_data = {
        "train_data": train_data,
        "test_data": test_data,
        "user_id_map": user_id_map,
        "movie_id_map": movie_id_map
    }
    
    # 确保 data 目录存在
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    # 保存数据
    pd.to_pickle(processed_data, PROCESSED_DATA_PATH)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

def check_data_quality(ratings):
    print("Checking data quality...")
    print(f"Number of NaN values: {ratings.isnull().sum().sum()}")
    print(f"Number of Inf values: {np.isinf(ratings.values).sum()}")
    print(f"Rating range: {ratings['rating'].min()} to {ratings['rating'].max()}")

if __name__ == "__main__":
    # 加载原始数据
    users, movies, ratings = load_data()
    # print(ratings)
    # 查看数据质量
    check_data_quality(ratings)
    
    # 预处理数据
    train_data, test_data, user_id_map, movie_id_map = preprocess_data(users, movies, ratings)
    print(train_data)
    # 保存预处理后的数据
    save_processed_data(train_data, test_data, user_id_map, movie_id_map)