# minimind/model/model_ltc_gollum.py

import torch
import torch.nn as nn
import numpy as np

# 咕噜LTC模型的配置类，模仿MiniMindConfig
class LTCGollumConfig:
    def __init__(self, input_size=5, hidden_size=32, output_size=4, **kwargs):
        self.input_size = input_size      # 环境刺激维度
        self.hidden_size = hidden_size    # 咕噜内在状态维度 (num_units)
        self.output_size = output_size    # 预测内在状态的维度

# LTC神经元细胞，是LTC模型的核心
class LTCCell(nn.Module):
    def __init__(self, config):
        super(LTCCell, self).__init__()
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.ode_solver_unfolds = 6  # 与原TF代码保持一致

        # --- 参数初始化：严格参考TF源码 ---
        # 这一步至关重要，好的初始化是成功训练的关键
        # 感觉神经元参数 (Sensory Neurons)
        self.sensory_mu = nn.Parameter(torch.rand(self.input_size, self.hidden_size) * 0.5 + 0.3)
        self.sensory_sigma = nn.Parameter(torch.rand(self.input_size, self.hidden_size) * 5.0 + 3.0)
        self.sensory_W = nn.Parameter(torch.rand(self.input_size, self.hidden_size) * 0.99 + 0.01)
        sensory_erev_init = torch.from_numpy(2 * np.random.randint(0, 2, size=[self.input_size, self.hidden_size]) - 1).float()
        self.sensory_erev = nn.Parameter(sensory_erev_init)

        # 内部神经元参数 (Internal Neurons)
        self.mu = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size) * 0.5 + 0.3)
        self.sigma = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size) * 5.0 + 3.0)
        self.W = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size) * 0.99 + 0.01)
        erev_init = torch.from_numpy(2 * np.random.randint(0, 2, size=[self.hidden_size, self.hidden_size]) - 1).float()
        self.erev = nn.Parameter(erev_init)
        
        # 泄漏和膜电容参数
        self.vleak = nn.Parameter(torch.rand(self.hidden_size) * 0.4 - 0.2)
        self.gleak = nn.Parameter(torch.ones(self.hidden_size))
        self.cm = nn.Parameter(torch.full((self.hidden_size,), 0.5))
    
    def _sigmoid(self, v_pre, mu, sigma):
        # TF版本中有一个reshape操作，我们在PyTorch中用unsqueeze实现同样效果
        # v_pre: (batch_size, hidden_size) -> (batch_size, hidden_size, 1)
        # mu/sigma: (hidden_size, hidden_size)
        v_pre = v_pre.unsqueeze(2)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def forward(self, inputs, state):
        """
        这是对TF代码 `_ode_step` (SemiImplicit solver)的PyTorch实现
        inputs shape: (batch_size, input_size)
        state shape: (batch_size, hidden_size)
        """
        v_pre = state

        # 1. 感觉输入处理 (Sensory input processing)
        # TF的_sigmoid实现，输入维度扩展为(batch_size, input_size, 1)以与参数进行广播
        sensory_x = inputs.unsqueeze(2) 
        sensory_mu_exp = self.sensory_mu.unsqueeze(0)
        sensory_sigma_exp = self.sensory_sigma.unsqueeze(0)
        
        sensory_activation = torch.sigmoid((sensory_x - sensory_mu_exp) * sensory_sigma_exp)
        
        sensory_w_activation = self.sensory_W * sensory_activation
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        
        # TF的reduce_sum(axis=1)对应PyTorch的sum(dim=1)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # 2. ODE求解器循环
        for _ in range(self.ode_solver_unfolds):
            # 内部神经元激活
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev
            
            # TF的reduce_sum(axis=1)对应PyTorch的sum(dim=1)
            # 注意: w_activation是3D的, rev_activation也是, sum应在dim=1上
            w_numerator_inter = torch.sum(rev_activation, dim=1)
            w_denominator_inter = torch.sum(w_activation, dim=1)

            w_numerator = w_numerator_inter + w_numerator_sensory
            w_denominator = w_denominator_inter + w_denominator_sensory
            
            numerator = self.cm * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm + self.gleak + w_denominator
            
            # 更新状态
            v_pre = numerator / (denominator + 1e-8)
        
        return v_pre

# 完整的咕噜模型，负责处理时间序列
class LTCGollum(nn.Module):
    def __init__(self, config):
        super(LTCGollum, self).__init__()
        self.config = config
        self.cell = LTCCell(config)
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)
    
    def forward(self, inputs):
        # inputs shape: (batch_size, sequence_length, input_size)
        batch_size = inputs.size(0)
        # 初始化隐藏状态
        hidden_state = torch.zeros(batch_size, self.config.hidden_size).to(inputs.device)
        
        outputs = []
        # 遍历时间序列
        for t in range(inputs.size(1)):
            hidden_state = self.cell(inputs[:, t, :], hidden_state)
            outputs.append(hidden_state)
            
        all_hidden_states = torch.stack(outputs, dim=1)
        predictions = self.output_layer(all_hidden_states)
        
        return predictions