import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from models.mlp_model import MLPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# 定义数据路径
PROCESSED_DATA_PATH = "data/processed_data.pkl"

# 定义超参数
'''BATCH_SIZE = 128
EMBEDDING_DIM = 32
HIDDEN_DIMS = [64, 32]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
NUM_EPOCHS = 10'''

# 定义超参数
BATCH_SIZE = 4096
EMBEDDING_DIM = 16
HIDDEN_DIMS = [32]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10

class RatingDataset(Dataset):
    def __init__(self, data):
        """
        自定义数据集类。
        
        参数：
        - data: 包含 user_id, movie_id 和 rating 的 DataFrame
        """
        self.user_ids = data["user_id"].values
        self.movie_ids = data["movie_id"].values
        self.ratings = data["rating"].values
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "movie_id": torch.tensor(self.movie_ids[idx], dtype=torch.long),
            "rating": torch.tensor(self.ratings[idx], dtype=torch.float32)
        }

def load_processed_data():
    """
    加载预处理后的数据。
    """
    processed_data = pd.read_pickle(PROCESSED_DATA_PATH)
    train_data = processed_data["train_data"]
    test_data = processed_data["test_data"]
    user_id_map = processed_data["user_id_map"]
    movie_id_map = processed_data["movie_id_map"]
    
    return train_data, test_data, len(user_id_map), len(movie_id_map)

def train():
    # 加载数据
    train_data, test_data, num_users, num_movies = load_processed_data()
    train_dataset = RatingDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型、损失函数和优化器
    model = MLPModel(num_users=num_users, num_movies=num_movies, embedding_dim=EMBEDDING_DIM,
                     hidden_dims=HIDDEN_DIMS, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.MSELoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 初始化优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    # 训练模型
    train_losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in progress_bar:
            user_ids = batch["user_id"].to(device)
            movie_ids = batch["movie_id"].to(device)
            ratings = batch["rating"].to(device)

            # 打印中间结果
            # print(f"User IDs: {user_ids}, Movie IDs: {movie_ids}, Ratings: {ratings}")
    
            # 前向传播
            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # 绘制损失曲线
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker="o")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()
    plt.savefig('Loss_emb16_2048.png')
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mlp_model_emb16_1024.pth")
    print("Model saved to models/mlp_model_emb16.pth")

if __name__ == "__main__":
    train()