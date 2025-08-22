# minimind/model/model_ltc_gollum_rl.py
# --- VERSION FOR REINFORCEMENT LEARNING (PHASE 2) ---

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

# -------------------------------------------------------------------
# Part 1: Core LTC Components (Re-used from Phase 1)
# -------------------------------------------------------------------

class LTCGollumConfig:
    """
    配置类，保持与阶段一兼容，但在RL中，output_size不再直接使用。
    """
    def __init__(self, input_size=5, hidden_size=32, output_size=4, **kwargs):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size # 主要用于兼容阶段一的config

class LTCCell(nn.Module):
    """
    LTC神经元细胞，与阶段一完全相同，是模型的核心计算单元。
    """
    def __init__(self, config):
        super(LTCCell, self).__init__()
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.ode_solver_unfolds = 6

        self.sensory_mu = nn.Parameter(torch.rand(self.input_size, self.hidden_size) * 0.5 + 0.3)
        self.sensory_sigma = nn.Parameter(torch.rand(self.input_size, self.hidden_size) * 5.0 + 3.0)
        self.sensory_W = nn.Parameter(torch.rand(self.input_size, self.hidden_size) * 0.99 + 0.01)
        sensory_erev_init = torch.from_numpy(2 * np.random.randint(0, 2, size=[self.input_size, self.hidden_size]) - 1).float()
        self.sensory_erev = nn.Parameter(sensory_erev_init)

        self.mu = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size) * 0.5 + 0.3)
        self.sigma = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size) * 5.0 + 3.0)
        self.W = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size) * 0.99 + 0.01)
        erev_init = torch.from_numpy(2 * np.random.randint(0, 2, size=[self.hidden_size, self.hidden_size]) - 1).float()
        self.erev = nn.Parameter(erev_init)
        
        self.vleak = nn.Parameter(torch.rand(self.hidden_size) * 0.4 - 0.2)
        self.gleak = nn.Parameter(torch.ones(self.hidden_size))
        self.cm = nn.Parameter(torch.full((self.hidden_size,), 0.5))
    
    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.unsqueeze(2)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def forward(self, inputs, state):
        v_pre = state
        sensory_x = inputs.unsqueeze(2) 
        sensory_mu_exp = self.sensory_mu.unsqueeze(0)
        sensory_sigma_exp = self.sensory_sigma.unsqueeze(0)
        sensory_activation = torch.sigmoid((sensory_x - sensory_mu_exp) * sensory_sigma_exp)
        sensory_w_activation = self.sensory_W * sensory_activation
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        for _ in range(self.ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev
            w_numerator_inter = torch.sum(rev_activation, dim=1)
            w_denominator_inter = torch.sum(w_activation, dim=1)
            w_numerator = w_numerator_inter + w_numerator_sensory
            w_denominator = w_denominator_inter + w_denominator_sensory
            numerator = self.cm * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm + self.gleak + w_denominator
            v_pre = numerator / (denominator + 1e-8)
        
        return v_pre

# -------------------------------------------------------------------
# Part 2: Specialized Models for Reinforcement Learning
# -------------------------------------------------------------------

class LTCGollum_FeatureExtractor(nn.Module):
    """
    一个专门的LTC模型版本，其唯一目的是作为特征提取器。
    它不包含最后的全连接输出层，直接返回LTC核心的高维隐藏状态。
    """
    def __init__(self, config):
        super(LTCGollum_FeatureExtractor, self).__init__()
        self.config = config
        self.cell = LTCCell(config)
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden_state = torch.zeros(batch_size, self.config.hidden_size).to(inputs.device)
        outputs = []
        for t in range(inputs.size(1)):
            hidden_state = self.cell(inputs[:, t, :], hidden_state)
            outputs.append(hidden_state)
        all_hidden_states = torch.stack(outputs, dim=1)
        # 直接返回高维隐藏状态
        return all_hidden_states

class ActorCriticLTC(nn.Module):
    """
    用于强化学习的Actor-Critic最终模型。
    它使用 LTCGollum_FeatureExtractor 作为共享的核心。
    """
    def __init__(self, config, num_actions):
        super(ActorCriticLTC, self).__init__()
        self.config = config
        self.num_actions = num_actions
        
        # 共享的LTC核心，现在使用专门的特征提取器版本
        self.ltc_core = LTCGollum_FeatureExtractor(config)
        
        # Actor Head (策略头)
        self.actor_head = nn.Linear(config.hidden_size, num_actions)
        
        # Critic Head (价值头)
        self.critic_head = nn.Linear(config.hidden_size, 1)

    def load_pretrained_core(self, pretrained_path):
        """
        从阶段一的模型加载权重。
        会自动忽略不匹配的 'output_layer' 权重。
        """
        try:
            device = next(self.parameters()).device
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            core_dict = self.ltc_core.state_dict()
            
            # 只加载在当前模型 (ltc_core) 中存在的键
            load_dict = {k: v for k, v in pretrained_dict.items() if k in core_dict and v.shape == core_dict[k].shape}
            
            core_dict.update(load_dict)
            self.ltc_core.load_state_dict(core_dict)
            print(f"Successfully loaded {len(load_dict)} matching layers into LTC feature extractor from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained core: {e}")

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        
        # ltc_output shape: (batch_size, sequence_length, hidden_size)
        ltc_output = self.ltc_core(inputs)
        # features shape: (batch_size, hidden_size)
        features = ltc_output[:, -1, :]
        
        action_logits = self.actor_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        state_value = self.critic_head(features)
        
        return action_probs, state_value

    def get_action_and_value(self, state, action=None):
        action_probs, state_value = self.forward(state)
        
        dist = Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, state_value