# minimind/eval_gollum_sl.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

# 确保模型和数据加载器代码与训练脚本一致
from model.model_ltc_gollum import LTCGollum, LTCGollumConfig
from trainer.train_gollum_sl import GollumDataset # 从我们的训练脚本中导入Dataset类

def evaluate_model(model, data_loader, device):
    """
    执行定量评估
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            predictions = model(inputs)
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 将所有批次的数据合并成一个大数组
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 为了使用sklearn metrics，我们需要将(样本数, 时间步, 特征数)的3D数组展平
    # 变成(样本数*时间步, 特征数)的2D数组
    num_samples, seq_len, num_features = all_labels.shape
    predictions_flat = all_predictions.reshape(-1, num_features)
    labels_flat = all_labels.reshape(-1, num_features)

    # --- 计算指标 ---
    mse = mean_squared_error(labels_flat, predictions_flat)
    mae = mean_absolute_error(labels_flat, predictions_flat)
    r2 = r2_score(labels_flat, predictions_flat)

    print("--- 咕噜本能核心 “体检报告” (定量) ---")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"决定系数 (R² Score): {r2:.6f}")
    print("------------------------------------------")
    
    # 返回未展平的数据用于可视化
    return all_predictions, all_labels

def visualize_predictions(predictions, labels, num_samples_to_plot=3):
    """
    执行定性评估 - 可视化
    """
    state_names = ['Energy', 'Safety', 'Satisfaction', 'Trust']
    num_sequences = predictions.shape[0]
    
    # 随机选择几个样本进行可视化
    sample_indices = np.random.choice(num_sequences, size=num_samples_to_plot, replace=False)
    
    print(f"--- 咕噜本能核心 “体检报告” (定性) ---")
    print(f"正在随机可视化 {num_samples_to_plot} 个生活片段...")

    for i, sample_idx in enumerate(sample_indices):
        # 创建一个包含4个子图的图表
        fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f'Gollum Life Snippet #{sample_idx}', fontsize=16)

        for state_idx in range(len(state_names)):
            ax = axs[state_idx]
            # 绘制真实值曲线
            ax.plot(labels[sample_idx, :, state_idx], label='Ground Truth (Teacher)', color='blue', linewidth=2)
            # 绘制预测值曲线
            ax.plot(predictions[sample_idx, :, state_idx], label='LTC Prediction (Student)', color='red', linestyle='--', linewidth=2)
            
            ax.set_ylabel(state_names[state_idx], fontsize=12)
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        axs[-1].set_xlabel('Time Steps', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def main():
    # --- 配置 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "checkpoints/gollum_sl/gollum_instinct_core.pth"
    val_data_path = 'dataset/gollum_dataset_val.npz'
    
    # --- 加载模型 ---
    print(f"正在从 {model_path} 加载模型...")
    config = LTCGollumConfig()
    model = LTCGollum(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("模型加载成功！")

    # --- 加载验证数据 ---
    val_dataset = GollumDataset(val_data_path)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --- 执行评估 ---
    predictions, labels = evaluate_model(model, val_loader, device)
    visualize_predictions(predictions, labels)

if __name__ == "__main__":
    main()