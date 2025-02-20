import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from models.mlp_model import MLPModel

# 定义数据路径
PROCESSED_DATA_PATH = "data/processed_data.pkl"
MODEL_PATH = "models/last_checkpoint.pth"
LAST_CHECKPOINT_PATH = "models/last_checkpoint.pth"

# 定义超参数
BATCH_SIZE = 1024
EMBEDDING_DIM = 32
HIDDEN_DIMS = [64, 32]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
CONTINUE_TRAINING = True  # 是否继续训练

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    
    # 如果继续训练，则加载上次的检查点
    if CONTINUE_TRAINING and os.path.exists(LAST_CHECKPOINT_PATH):
        checkpoint = torch.load(LAST_CHECKPOINT_PATH)
        # 打印检查点内容
        print("Keys in checkpoint:", checkpoint.keys())
        for key, value in checkpoint.items():
            print(f"{key}: {type(value)}")
        model.load_state_dict(checkpoint['model_state_dict'])
        # 初始化优化器（即使加载失败也需要初始化）
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        train_losses = []
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 定义损失函数
    criterion = nn.MSELoss().to(device)
    
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    # 训练模型
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + NUM_EPOCHS}")
        for batch in progress_bar:
            user_ids = batch["user_id"].to(device)  # 将数据移动到 GPU
            movie_ids = batch["movie_id"].to(device)
            ratings = batch["rating"].to(device)
            
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
        
        # 更新学习率调度器
        scheduler.step(avg_loss)
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses
        }
        torch.save(checkpoint, LAST_CHECKPOINT_PATH)
        print(f"Checkpoint saved at epoch {epoch+1}")
    
    # 绘制损失曲线
    #plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker="o")
    #plt.title("Training Loss Curve")
    #plt.xlabel("Epoch")
    #plt.ylabel("Loss")
    #plt.grid()
    #plt.show()
    #plt.savefig(f"last_checkpoint_{start_epoch+10}.png")
    
    # 保存模型
    torch.save(checkpoint, "models/last_checkpoint.pth")
    print("Model saved to models/last_checkpoint.pth")

if __name__ == "__main__":
    train()