# minimind/trainer/train_gollum_sl.py

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_ltc_gollum import LTCGollum, LTCGollumConfig
import time

# --- 1. 日志记录器 (从MiniMind借鉴) ---
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

# --- 2. 数据集类 ---
# 正确的
class GollumDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        # 使用我们保存时定义的键 'inputs' 和 'labels'
        self.inputs = torch.from_numpy(data['inputs']).float()
        self.labels = torch.from_numpy(data['labels']).float()
        print(f"Loaded data from {file_path}. Input shape: {self.inputs.shape}, Label shape: {self.labels.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# --- 3. 训练主函数 ---
def main():
    # --- 配置参数 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 64
    learning_rate = 1e-3
    output_dir = "checkpoints/gollum_sl"
    log_file = os.path.join(output_dir, "train_log.txt")
    os.makedirs(output_dir, exist_ok=True)
    logger = Logger(log_file)

    logger.log(f"Using device: {device}")

    # --- 初始化模型 ---
    logger.log("Initializing model...")
    config = LTCGollumConfig()
    model = LTCGollum(config)
    model.to(device)
    # 检查PyTorch版本后，启用编译
    # if torch.__version__[0] == '2':
    #     logger.log("PyTorch 2.0+ detected, compiling the model...")
    #     model = torch.compile(model)
        
    logger.log(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")
        # --- 准备数据 ---
    logger.log("Loading dataset...")
    train_dataset = GollumDataset('dataset/gollum_dataset_train.npz')
    val_dataset = GollumDataset('dataset/gollum_dataset_val.npz')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # --- 定义优化器和损失函数 ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # --- 训练循环 ---
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(inputs)
            loss_weights = torch.tensor([2.0, 1.0, 2.0, 2.0], device=device) # [Energy, Safety, Satisfaction, Trust]
            # 给Energy, Satisfaction, Trust两倍的权重，Safety保持原权重

            # (predictions - labels)**2 得到每个元素的平方误差
            squared_errors = (predictions - labels) ** 2
            # 乘以权重
            weighted_squared_errors = squared_errors * loss_weights
            # 求平均得到最终损失
            loss = torch.mean(weighted_squared_errors)

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 验证 ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        logger.log(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.2f}s")
        
        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(output_dir, "gollum_instinct_core.pth")
            torch.save(model.state_dict(), save_path)
            logger.log(f"New best model saved to {save_path} with val loss {best_val_loss:.6f}")

    logger.log("Training finished.")

if __name__ == "__main__":
    main()