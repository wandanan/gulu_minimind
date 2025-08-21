# minimind/dataset/gollum_data_generator.py

import numpy as np
import os
from tqdm import tqdm

# --- 配置 ---
CONFIG = {
    "TOTAL_SAMPLES": 1200,          # 总共生成的样本数量
    "TRAIN_RATIO": 0.8,             # 训练集所占比例
    "SEQUENCE_LENGTH": 200,         # 每个样本（生活录像）的长度
    "INPUT_DIM": 5,                 # 输入维度 (环境刺激)
    "STATE_DIM": 4,                 # 内部状态维度 (咕噜的情绪)
    "DATASET_DIR": "./dataset",       # 数据集保存目录
    "TRAIN_FILENAME": "gollum_dataset_train.npy",
    "VAL_FILENAME": "gollum_dataset_val.npy",
}

def generate_one_sequence():
    """
    生成一个独立的咕噜“生活录像”片段。
    这个函数是咕噜世界的“物理引擎”和“剧本导演”。
    """
    # 初始化输入和状态矩阵
    # inputs: [player_presence, toy_presence, threat_presence, player_gives_toy, player_takes_toy]
    inputs = np.zeros((CONFIG["SEQUENCE_LENGTH"], CONFIG["INPUT_DIM"]))
    # states: [energy, safety, curiosity, trust]
    states = np.zeros((CONFIG["SEQUENCE_LENGTH"], CONFIG["STATE_DIM"]))
    
    # 随机初始化咕噜的初始内在状态
    states[0, :] = np.random.rand(CONFIG["STATE_DIM"]) * 0.5 + 0.25

    # 模拟事件的持续时间
    player_timer = 0
    toy_timer = 0
    threat_timer = 0

    # 逐个时间步进行模拟
    for t in range(1, CONFIG["SEQUENCE_LENGTH"]):
        # --- 1. 生成当前时间步的环境刺激 (剧本) ---
        # 随机事件发生器
        if player_timer <= 0 and np.random.rand() < 0.05:
            player_timer = np.random.randint(20, 50) # 玩家出现一段时间
        if toy_timer <= 0 and np.random.rand() < 0.03:
            toy_timer = np.random.randint(15, 40)   # 玩具出现一段时间
        if threat_timer <= 0 and np.random.rand() < 0.02:
            threat_timer = np.random.randint(10, 20)  # 威胁出现一段时间

        # 更新当前刺激
        player_presence = 1 if player_timer > 0 else 0
        toy_presence = 1 if toy_timer > 0 else 0
        threat_presence = 1 if threat_timer > 0 else 0
        
        # 模拟玩家互动事件 (给予/拿走玩具)
        player_gives_toy = 1 if player_presence and toy_presence and np.random.rand() < 0.1 else 0
        player_takes_toy = 1 if player_presence and toy_presence and np.random.rand() < 0.05 else 0

        inputs[t, :] = [player_presence, toy_presence, threat_presence, player_gives_toy, player_takes_toy]
        
        # 更新计时器
        player_timer -= 1
        toy_timer -= 1
        threat_timer -= 1

        # --- 2. 计算内在状态的变化 (物理引擎) ---
        prev_state = states[t-1, :]
        energy, safety, curiosity, trust = prev_state

        # 定义每个状态的目标值 (Target) 和时间常数 (Tau)
        # 这是咕噜的“情绪物理学规则”
        
        # 能量 (Energy)
        target_energy = 0.5 - (player_presence + toy_presence) * 0.2 # 活动消耗能量
        tau_energy = 50.0 # 能量变化缓慢
        
        # 安全感 (Safety)
        target_safety = 0.8 - threat_presence * 1.0 + player_presence * (trust - 0.5) * 0.4
        tau_safety = 10.0 # 受到威胁时反应快
        
        # 好奇心 (Curiosity)
        target_curiosity = 0.1 + toy_presence * 0.8 * safety - curiosity * 0.1 # 只有在安全时才会对玩具有好奇心
        tau_curiosity = 5.0 + toy_presence * 20.0 # 对玩具的兴趣会随时间慢慢减弱
        
        # 信任度 (Trust)
        target_trust = trust + player_gives_toy * 0.1 - player_takes_toy * 0.2 - threat_presence * player_presence * 0.1
        tau_trust = 100.0 # 信任是长期建立的
        
        targets = np.array([target_energy, target_safety, target_curiosity, target_trust])
        taus = np.array([tau_energy, tau_safety, tau_curiosity, tau_trust])
        
        # 使用ODE的离散形式（欧拉法）更新状态
        d_state = (targets - prev_state) / taus
        new_state = prev_state + d_state
        
        # 将状态限制在 [0, 1] 区间
        states[t, :] = np.clip(new_state, 0, 1)

    return inputs, states


def generate_and_save_data():
    """
    主函数，生成所有数据并保存到文件。
    """
    print("--- Gollum Data Generator ---")
    
    num_train = int(CONFIG["TOTAL_SAMPLES"] * CONFIG["TRAIN_RATIO"])
    num_val = CONFIG["TOTAL_SAMPLES"] - num_train
    
    print(f"Total samples: {CONFIG['TOTAL_SAMPLES']}")
    print(f"Training samples: {num_train}")
    print(f"Validation samples: {num_val}")

    # --- 生成训练集 ---
    print("\nGenerating training data...")
    train_inputs, train_labels = [], []
    for _ in tqdm(range(num_train)):
        i, s = generate_one_sequence()
        train_inputs.append(i)
        train_labels.append(s)
    
    # 组合成最终的数据格式
    train_data = {
        'inputs': np.array(train_inputs),
        'labels': np.array(train_labels)
    }


    # --- 生成验证集 ---
    print("\nGenerating validation data...")
    val_inputs, val_labels = [], []
    for _ in tqdm(range(num_val)):
        i, s = generate_one_sequence()
        val_inputs.append(i)
        val_labels.append(s)

    val_data = {
        'inputs': np.array(val_inputs),
        'labels': np.array(val_labels)
    }


    # --- 保存文件 ---
    os.makedirs(CONFIG["DATASET_DIR"], exist_ok=True)
    # 为了清晰，我们把文件名后缀改为 .npz
    train_filename_npz = CONFIG["TRAIN_FILENAME"].replace('.npy', '.npz')
    val_filename_npz = CONFIG["VAL_FILENAME"].replace('.npy', '.npz')
    train_path = os.path.join(CONFIG["DATASET_DIR"], train_filename_npz)
    val_path = os.path.join(CONFIG["DATASET_DIR"], val_filename_npz)

    print(f"\nSaving training data to {train_path}...")
    # 使用 np.savez 并直接传入关键字参数，这会自动使用字典的键作为数组名
    np.savez(train_path, inputs=train_data['inputs'], labels=train_data['labels'])
    print(f"  -> Shape: inputs {train_data['inputs'].shape}, labels {train_data['labels'].shape}")

    print(f"Saving validation data to {val_path}...")
    np.savez(val_path, inputs=val_data['inputs'], labels=val_data['labels'])
    print(f"  -> Shape: inputs {val_data['inputs'].shape}, labels {val_data['labels'].shape}")
if __name__ == "__main__":
    generate_and_save_data()