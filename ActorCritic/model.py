from copy import copy, deepcopy
import numpy as np
from numpy import ndarray

import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
from torch.nn import functional as F

from typing import List, Tuple, Type, Callable

from ReplayBuffer.Buffer import ReplayBuffer

from usefulParam.Param import *

from mutil_RL.mutil_torch import factory_LinearReLU_ModuleList, factory_LinearReLU_Sequential, conv_str2Optimizer, hard_update, soft_update


class ContinuousActorNet(nn.Module):
    def __init__(self, in_chnls, hdn_chnls, hdn_lays, out_chnls):
        super().__init__()
        self._in_chnls = in_chnls
        self._hdn_chnls = hdn_chnls
        self._hdn_lays = hdn_lays
        self._out_chnls = out_chnls

        self._laysList = nn.ModuleList()

        # 入力層
        self._laysList.append(nn.Linear(self._in_chnls, self._hdn_chnls))
        self._laysList.append(nn.ReLU())

        # 隠れ層
        for _ in range(self._hdn_lays):
            self._laysList.append(nn.Linear(self._hdn_chnls, self._hdn_chnls))
            self._laysList.append(nn.ReLU())
        
        # 出力層
        self._mean_layer = nn.Linear(self._hdn_chnls, self._out_chnls)
        self._std_layer = nn.Linear(self._hdn_chnls, self._out_chnls)
    
    def forward(self, status: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 入力調整
        if len(status.shape) == 1:
            status = status.unsqueeze(0)
        
        for layer in self._laysList:
            status = layer(status)
        mean = self._mean_layer(status)
        std = F.softplus(self._std_layer(status)) + 1e-6

        return mean, std

class CriticNet(nn.Module):
    '''
        Q値推定 Critic
    '''

    def __init__(self, 
                 action_size: int, state_size: int, 
                 hdn_chnls: int, hdn_layers: int, out_chnls: int):
        super().__init__()

        # 変数定義
        self._action_size = action_size
        self._state_size = state_size
        self._in_chnls = self._action_size + self._state_size
        self._hdn_chnls = hdn_chnls
        self._hdn_layers = hdn_layers
        self._out_chnls = out_chnls

        # ネットワーク定義
        self._network = factory_LinearReLU_Sequential(self._in_chnls, self._hdn_chnls, self._hdn_layers, self._out_chnls)
    
    def forward(self, status: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # 入力調整
        if len(status.shape) == 1:
            status = status.unsqueeze(1)

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)
        # comm
        x = torch.cat([status, actions], dim=1)
        x = self._network.forward(x)
        
        return x

class ActorCriticAgent:
    def __init__(self, gamma: ScalarParam, lr: ScalarParam, tau: ScalarParam, 
                 state_size: int, action_size: int, # net common param
                 actor_hdn_chnls: int, actor_hdn_lays: int, actor_potimizer: str,    # Actor
                 critic_hdn_chnls: int, critic_hdn_lays: int, critic_optimizer: str,  # Critic
                 critic_sync_interval: int, # net_sync
                 buf_capacity: int, batch_size: int, # replayBuf
                 device: torch.device # device
                 ):
        # デバイス 決定
        self._device = device

        # ハイパーパラメータ
        self._gamma = gamma
        self._lr = lr
        self._tau = tau

        # Actor net
        self._actor_net = ContinuousActorNet(state_size, actor_hdn_chnls, actor_hdn_lays, action_size)
        self._actor_optimizer = conv_str2Optimizer(critic_optimizer, self._actor_net.parameters(), lr=self._lr.value)

        # Critic net
        self._critic_net = CriticNet(action_size, state_size, critic_hdn_chnls, critic_hdn_lays, 1)
        self._critic_net_target = deepcopy(self._critic_net)
        self._critic_optimizer = conv_str2Optimizer(critic_optimizer, self._critic_net.parameters(), lr=self._lr.value)
        self._critic_loss = nn.MSELoss()
        self._interval_count = 0
        self._critic_sync_interval = critic_sync_interval

        # リプレイバッファ
        self._replayBuf = ReplayBuffer(buf_capacity, state_size, action_size, device=self._device)
        self._batch_size = batch_size

        # ログ
        self._actor_loss_history = list()
        self._critic_loss_history = list()

    def _get_action_comm(self, status: torch.Tensor) -> torch.Tensor:
        '''
            行動の選択
            
        '''
        # 入力調整
        if not type(status) == torch.Tensor:
            status = torch.tensor(status, device=self._device)

        means, stds = self._actor_net.forward(status)
        normal = Normal(means, stds)
        actions = torch.tanh(normal.rsample()) * 2.0
        return actions.squeeze(0)
        
    
    def get_action_np(self, status: torch.Tensor) -> np.ndarray:
        '''
            行動の選択を行い，np.ndarrayで返却する
        '''
        actions = self._get_action_comm(status)
        return actions.detach().numpy()

    def get_action_torch(self, status: torch.Tensor) -> torch.Tensor:
        return self._get_action_comm(status)
    
    def get_action_from_meanstd_torch(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)
        
        if len(std.shape) == 1:
            std = std.unsqueeze(0)
        
        normal = Normal(mean, std)
        actions = torch.tanh(normal.rsample()) * 2.0

        return actions
    
    def get_action_from_meanstd_np(self, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
        return self.get_action_from_meanstd_torch(mean, std).detach().numpy()
    
    def update(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray):
        '''
            リプレイバッファの更新
            ネットワークの更新
        '''
        ### リプレイバッファの更新
        self.update_repbuf(state, action, reward, next_state, done)

        ### ネットワークの更新
        if not self.do_update_network():
            return 

        self.update_network()
    
    def update_repbuf(self, state, action, reward, next_state, done):
        '''
        リプレイバッファの更新
        '''
        self._replayBuf.add(state, action, reward, next_state, done)
    
    def update_network(self):
        '''
        ネットワークの更新
        '''
        ## バッチの取得
        batch = self._replayBuf.get_batch(self._batch_size)
        status, actions, rewards, next_status, dones = batch

        ## Criticの更新
        next_action4update = self.get_action_from_meanstd_torch(*self._actor_net.forward(next_status))
        q_target = rewards + self._gamma.value * self._critic_net_target(next_status, next_action4update) * (1 - dones)
        q_val = self._critic_net(status, actions)
        critic_loss = self._critic_loss(q_val, q_target.detach())

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        ## Actorの更新
        action4update = self.get_action_from_meanstd_torch(*self._actor_net.forward(status))
        actor_loss = -self._critic_net.forward(status, action4update).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        ## critic_netの同期処理
        self._interval_count, do_sync = self.step_interval_count(self._interval_count)
        if do_sync:
            soft_update(self._critic_net, self._critic_net_target, self._tau.value)
        
        ## 記録
        self.logging_actor_loss(actor_loss)
        self.logging_critic_loss(critic_loss)
    
    def do_update_network(self) -> bool:
        '''
        ネットワークの更新を行うか
        '''
        return self._replayBuf.real_size() >= self._batch_size

    
    def step_interval_count(self, interval_count: int) -> Tuple[int, bool]:
        interval_count = (interval_count + 1) % self._critic_sync_interval
        do_sync = (interval_count == 0)
        return interval_count, do_sync
    
    def logging_actor_loss(self, actor_loss) -> None:
        self.actor_loss_history.append(actor_loss)
    
    def logging_critic_loss(self, critic_loss) -> None:
        self._critic_loss_history.append(critic_loss)

    @property
    def actor_loss_history(self):
        return self._actor_loss_history
    
    @property
    def critic_loss_history(self):
        return self._critic_loss_history