import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
from models.eges import EGES
from torch.utils.data import DataLoader, Dataset
from utils.graph_builder import build_graph

# 超参数
EMBEDDING_DIM = 64
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

def train():
    # 加载预处理后的数据
    processed_data = pd.read_pickle("data/processed_data.pkl")
    train_data = processed_data["train_data"]
    test_data = processed_data["test_data"]
    user_id_map = processed_data["user_id_map"]
    movie_id_map = processed_data["movie_id_map"]
    
    # 构建图
    users = pd.read_csv("data/ml-1m/users.dat", sep="::", engine="python", header=None, names=["user_id", "gender", "age", "occupation", "zip_code"])
    movies = pd.read_csv("data/ml-1m/movies.dat", sep="::", engine="python", header=None, names=["movie_id", "title", "genres"], encoding="ISO-8859-1")
    graph = build_graph(train_data, users, movies)
    
    # 初始化模型
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)
    num_side_info = 100  # 示例值，根据实际侧信息数量调整
    model = EGES(num_users, num_movies, EMBEDDING_DIM, num_side_info)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        # 遍历训练数据
        for _, batch in tqdm(train_data.groupby(np.arange(len(train_data)) // BATCH_SIZE)):
            user_ids = torch.tensor(batch["user_id"].values, dtype=torch.long)
            movie_ids = torch.tensor(batch["movie_id"].values, dtype=torch.long)
            ratings = torch.tensor(batch["rating"].values, dtype=torch.float32)
            
            # 假设侧信息 ID 已经准备好
            side_info_ids = torch.randint(0, num_side_info, (len(batch),), dtype=torch.long)
            
            # 前向传播
            predictions = model(user_ids, movie_ids, side_info_ids)
            loss = criterion(predictions, ratings)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss / len(train_data)}")
    
    # 保存模型
    torch.save(model.state_dict(), "models/eges.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()