import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.youtube_dnn import YouTubeDNN

# 定义数据路径
PROCESSED_DATA_PATH = "data/processed_data.pkl"
MODEL_PATH = "models/youtube_dnn.pth"

# 定义超参数
BATCH_SIZE = 128
EMBEDDING_DIM = 32
HIDDEN_DIMS = [64, 32]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class RatingDataset(Dataset):
    def __init__(self, data):
        """
        自定义数据集类。
        
        参数：
        - data: 包含 user_id 和 movie_id 的 DataFrame
        """
        self.user_ids = data["user_id"].values
        self.movie_ids = data["movie_id"].values
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "movie_id": torch.tensor(self.movie_ids[idx], dtype=torch.long)
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
    train_data, _, num_users, num_movies = load_processed_data()
    train_dataset = RatingDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型、损失函数和优化器
    model = YouTubeDNN(num_users=num_users, num_movies=num_movies, embedding_dim=EMBEDDING_DIM,
                       hidden_dims=HIDDEN_DIMS, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in progress_bar:
            user_ids = batch["user_id"].to(device)
            movie_ids = batch["movie_id"].to(device)
            
            # 前向传播
            logits = model(user_ids)
            loss = criterion(logits, movie_ids)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()