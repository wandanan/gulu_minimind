# minimind/trainer/train_gollum_rl_production_ppo.py
# --- FINAL & COMPLETE VERSION ---

import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environments.gollum_env import GollumEnv
from model.model_ltc_gollum_rl import LTCGollumConfig, ActorCriticLTC

# =======================================================
# Part 1: Helper function for multiprocessing
# =======================================================

def make_env():
    """
    一个可被pickle的顶级函数，用于创建GollumEnv实例。
    这是为了兼容Windows的'spawn'多进程启动方法。
    """
    return GollumEnv()

# =======================================================
# Part 2: Vectorized Environment Wrapper
# =======================================================

def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    obs, info = env.reset()
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == 'reset':
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
        except EOFError:
            break

class SubprocVecEnv:
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(nenvs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, env_fn))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, terminateds, truncateds, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(terminateds), np.stack(truncateds), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
        
    def get_spaces(self):
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        return observation_space, action_space

    @property
    def num_envs(self):
        return len(self.remotes)

# =======================================================
# Part 3: The PPO Trainer Class
# =======================================================

class PPO_Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.writer = SummaryWriter(log_dir=self.config['log_dir'])
        print(f"TensorBoard logs will be saved to: {self.config['log_dir']}")

        print("Creating subprocess vectorized environments...")
        env_fns = [make_env for _ in range(self.config['num_envs'])]
        self.envs = SubprocVecEnv(env_fns)
        print(f"{self.config['num_envs']} parallel environments created.")
        
        obs_space, act_space = self.envs.get_spaces()
        self.obs_shape = obs_space.shape
        self.num_actions = act_space.n
        print(f"Environment specs: num_actions={self.num_actions}, obs_shape={self.obs_shape}")
        
        ltc_config = LTCGollumConfig(input_size=self.obs_shape[0])
        # 模型创建和移动到设备，没有JIT
        self.model = ActorCriticLTC(ltc_config, num_actions=self.num_actions).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], eps=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))

        if self.config['pretrained_path'] and os.path.exists(self.config['pretrained_path']):
            self.model.load_pretrained_core(self.config['pretrained_path'])
        else:
            print("Warning: Pretrained model not found or path not specified. Training from scratch.")

    def compute_advantages_gae(self, rewards, values, dones, gamma, gae_lambda):
        num_steps = rewards.shape[0]
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(num_steps)):
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            advantages[t] = last_advantage
            
        returns = advantages + values[:-1]
        return advantages, returns

    def train(self):
        print(f"--- Starting PPO Training on {self.device} ---")
        start_time = time.time()
        
        # === 存储数组初始化 (保持不变) ===
        obs_shape = (self.config['num_steps_per_collect'], self.config['num_envs']) + self.obs_shape
        obs_np = np.zeros(obs_shape, dtype=np.float32)
        actions_np = np.zeros((self.config['num_steps_per_collect'], self.config['num_envs']), dtype=np.int64)
        log_probs_np = np.zeros((self.config['num_steps_per_collect'], self.config['num_envs']), dtype=np.float32)
        rewards_np = np.zeros((self.config['num_steps_per_collect'], self.config['num_envs']), dtype=np.float32)
        dones_np = np.zeros((self.config['num_steps_per_collect'], self.config['num_envs']), dtype=np.float32)
        values_np = np.zeros((self.config['num_steps_per_collect'] + 1, self.config['num_envs']), dtype=np.float32)
        # =================================================

        global_step = 0
        num_updates = self.config['total_timesteps'] // (self.config['num_steps_per_collect'] * self.config['num_envs'])
        
        print("Initial reset of environments...")
        next_obs, _ = self.envs.reset()
        next_done = np.zeros(self.config['num_envs'])
        print("Initial reset complete. Starting training loop...")

        for update in range(1, num_updates + 1):
            print(f"\n--- Update {update}/{num_updates} ---")
            collection_start_time = time.time()
            
            # 1. 数据收集 (与之前相同)
            for step in range(self.config['num_steps_per_collect']):
                global_step += 1 * self.config['num_envs']
                obs_np[step] = next_obs
                dones_np[step] = next_done
                
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(next_obs).to(self.device)
                    action, log_prob, value = self.model.get_action_and_value(obs_tensor)
                
                values_np[step] = value.cpu().numpy().flatten()
                actions_np[step] = action.cpu().numpy()
                log_probs_np[step] = log_prob.cpu().numpy()

                self.envs.step_async(action.cpu().numpy())
                next_obs, reward, terminated, truncated, infos = self.envs.step_wait()
                next_done = np.logical_or(terminated, truncated)
                rewards_np[step] = reward
            
            collection_end_time = time.time()
            
            # 2. 计算最后一个状态的价值 (与之前相同)
            with torch.no_grad():
                next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
                _, _, next_value = self.model.get_action_and_value(next_obs_tensor)
                values_np[-1] = next_value.cpu().numpy().flatten()
            
            # 3. 计算优势和回报 (与之前相同)
            advantages, returns = self.compute_advantages_gae(rewards_np, values_np, dones_np, self.config['gamma'], self.config['gae_lambda'])
            
            # === OPTIMIZATION 1: 一次性数据准备和上传 ===
            # 将所有Numpy数组扁平化
            b_obs = obs_np.reshape((-1,) + self.obs_shape)
            b_log_probs = log_probs_np.flatten()
            b_actions = actions_np.flatten()
            b_advantages = advantages.flatten()
            b_returns = returns.flatten()

            # 归一化优势
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            # 一次性将所有数据上传到GPU
            try:
                b_obs_gpu = torch.from_numpy(b_obs).to(self.device)
                b_actions_gpu = torch.from_numpy(b_actions).to(self.device, dtype=torch.long)
                b_log_probs_gpu = torch.from_numpy(b_log_probs).to(self.device)
                b_advantages_gpu = torch.from_numpy(b_advantages).to(self.device)
                b_returns_gpu = torch.from_numpy(b_returns).to(self.device)
            except Exception as e:
                print(f"Error during data upload to GPU: {e}")
                continue # Skip this update if something goes wrong
            # ===============================================

            update_start_time = time.time()
            
            b_indices = np.arange(len(b_obs))
            for epoch in range(self.config['num_updates_per_collect']):
                np.random.shuffle(b_indices)
                for start in range(0, len(b_obs), self.config['batch_size']):
                    end = start + self.config['batch_size']
                    mb_indices = b_indices[start:end]
                    
                    # === OPTIMIZATION 2: 直接从GPU Tensor切片 ===
                    mb_obs = b_obs_gpu[mb_indices]
                    mb_actions = b_actions_gpu[mb_indices]
                    mb_log_probs = b_log_probs_gpu[mb_indices]
                    mb_advantages = b_advantages_gpu[mb_indices]
                    mb_returns = b_returns_gpu[mb_indices]
                    # ===============================================

                    # 自动混合精度上下文
                    with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                        action_probs, state_values = self.model(mb_obs)
                        dist = Categorical(action_probs)
                        new_log_probs = dist.log_prob(mb_actions)
                        entropy = dist.entropy()
                        
                        ratio = torch.exp(new_log_probs - mb_log_probs)

                        surr1 = ratio * mb_advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.config['clip_epsilon'], 1.0 + self.config['clip_epsilon']) * mb_advantages
                        actor_loss = -torch.min(surr1, surr2).mean()

                        # 确保state_values是正确的形状以进行广播
                        state_values = state_values.squeeze()
                        if state_values.shape != mb_returns.shape:
                           # 临时调试信息
                           # print(f"Shape mismatch: returns {mb_returns.shape}, values {state_values.shape}")
                           state_values = state_values.view_as(mb_returns)

                        critic_loss = (mb_returns - state_values).pow(2).mean()
                        loss = actor_loss + self.config['value_coef'] * critic_loss - self.config['entropy_coef'] * entropy.mean()

                    # 梯度更新
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            
            update_end_time = time.time()

            # 日志记录
            avg_reward_per_episode = np.sum(rewards_np) / (np.sum(dones_np) + 1e-8)
            self.writer.add_scalar("charts/avg_episode_reward", avg_reward_per_episode, global_step)
            self.writer.add_scalar("losses/value_loss", critic_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", actor_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
            
            sps = int(self.config['num_steps_per_collect'] * self.config['num_envs'] / (update_end_time - collection_start_time))
            print(f"Global Step: {global_step}, SPS: {sps}, Avg Reward/Ep: {avg_reward_per_episode:.2f}, Update Time: {update_end_time-update_start_time:.2f}s, Collect Time: {collection_end_time-collection_start_time:.2f}s")

            # 模型保存
            if update % self.config['save_freq_updates'] == 0:
                save_path = os.path.join(self.config['output_dir'], f"gollum_policy_step_{global_step}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
        
        self.envs.close()
        self.writer.close()
        print("--- PPO Training Finished ---")

# =======================================================
# Part 4: Main execution block
# =======================================================

if __name__ == "__main__":
    config = {
        "num_envs": 20,
        "total_timesteps": 5_000_000,
        "num_steps_per_collect": 2048,
        "num_updates_per_collect": 10,
        "batch_size": 64,
        "learning_rate": 2.5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "log_dir": "runs/gollum_ppo_prod",
        "output_dir": "checkpoints/gollum_ppo_prod",
        "save_freq_updates": 50,
        "pretrained_path": "checkpoints/gollum_sl/gollum_instinct_core.pth"
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    if sys.platform != 'win32':
        if 'forkserver' in mp.get_all_start_methods():
            mp.set_start_method("forkserver", force=True)
            print("INFO: Using 'forkserver' for multiprocessing (Linux/macOS optimized).")
        else:
            mp.set_start_method("fork", force=True)
            print("INFO: 'forkserver' not available, using 'fork' for multiprocessing.")
    else:
        print("INFO: Using default 'spawn' for multiprocessing (Windows compatible).")

    trainer = PPO_Trainer(config)
    trainer.train()