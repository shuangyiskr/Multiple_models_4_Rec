import os
import pandas as pd
from tqdm import tqdm
from models.latent_factor_model import LatentFactorModel

# 定义数据路径
PROCESSED_DATA_PATH = "data/processed_data.pkl"
MODEL_PATH = "models/lfm_model.pkl"

def load_processed_data():
    """
    加载预处理后的数据。
    """
    processed_data = pd.read_pickle(PROCESSED_DATA_PATH)
    train_data = processed_data["train_data"]
    return train_data

def train():
    # 加载数据
    train_data = load_processed_data()
    
    # 初始化隐语义模型
    lfm_model = LatentFactorModel(num_factors=32)
    
    # 训练模型
    print("Training Latent Factor Model...")
    lfm_model.fit(train_data)
    print("Model training completed.")
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    import pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(lfm_model, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()