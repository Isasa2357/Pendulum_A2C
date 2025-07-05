from copy import copy, deepcopy

import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
from torch.nn import functional as F

from typing import List, Tuple, Type, Callable

from ReplayBuffer.Buffer import ReplayBuffer

from usefulParam.ScalarParam import *

from mutil_RL.mutil_torch import factory_LinearReLU_ModuleList


class ContinuousActorNet(nn.Module):
    def __init__(self, in_chnls, hdn_chnls, hdn_lays, out_chnls):

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
                 action_in_chnls: int, action_hdn_chnls: int, action_hdn_layers: int, action_out_chnls: int, 
                 state_in_chnls: int, state_hdn_chnls: int, state_hdn_layers: int, state_out_chnls: int, 
                 comm_hdn_chnls: int, comm_hdn_layers: int):
        # action特徴抽出 変数
        self._action_in_chnls = action_in_chnls
        self._action_hdn_chnls = action_hdn_chnls
        self._action_hdn_layers = action_hdn_layers
        self._action_out_chnls = action_out_chnls

        # state 特徴抽出 変数
        self._state_in_chnls = state_in_chnls
        self._state_hdn_chnls = state_hdn_chnls
        self._state_hdn_layers = state_hdn_layers
        self._state_out_chnls = state_out_chnls

        # actionとstateの統合部分 変数
        self._comm_in_chnls = action_out_chnls + state_out_chnls
        self._comm_hdn_chnls = comm_hdn_chnls
        self._comm_hdn_layers = comm_hdn_layers

        self._action_laysList = factory_LinearReLU_ModuleList(self._action_in_chnls, 
                                                             self._action_hdn_chnls, 
                                                             self._action_hdn_layers, 
                                                             self._action_out_chnls, 
                                                             "ReLU")
        self._state_laysList = factory_LinearReLU_ModuleList(self._state_in_chnls, 
                                                             self._state_hdn_chnls, 
                                                             self._state_hdn_layers, 
                                                             self._state_out_chnls,
                                                             "ReLU")
        self._comm_laysList = factory_LinearReLU_ModuleList(self._comm_in_chnls, 
                                                            self._comm_hdn_chnls, 
                                                            self._comm_hdn_layers, 
                                                            1, 
                                                            "ReLU")

    
    def forward(self, status: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # 入力調整
        if len(status.shape) == 1:
            status = status.unsqueeze(0)

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        # action
        for layer in self._action_laysList:
            actions = layer.forward(actions)
        
        # state
        for layer in self._state_laysList:
            status = layer.forward(status)

        # comm
        x = torch.cat([status, actions], dim=1)
        for layer in self._comm_laysList:
            x = layer.forward(x)
        
        return x

class ActorCriticAgent:
    def __init__(self, gamma: ScalarParam, lr: ScalarParam, tau: ScalarParam, 
                 state_size: int, action_size: int, # net common param
                 actor_hdn_chnls: int, actor_hdn_lays: int, actor_potimizer: Callable[..., optim.Optimizer],    # Actor
                 critic_hdn_chnls: int, critic_hdn_lays: int, critic_optimizer: Callable[..., optim.Optimizer],  # Critic
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
        self._actor_optimizer = actor_potimizer(self._actor_net.parameters(), lr=self._lr)

        # Critic net
        self._critic_net = CriticNet(action_size, 16, 1, 16, 
                                     state_size, 64, 2, 64, 
                                     critic_hdn_chnls, critic_hdn_lays)
        self._critic_net_target = deepcopy(self._critic_net)
        self._critic_optimizer = critic_optimizer(self._critic_net.parameters(), lr=self._lr)
        self._critic_loss = nn.MSELoss()
        self._interval_count = 0
        self._critic_sync_interval = critic_sync_interval

        # リプレイバッファ
        self._replayBuf = ReplayBuffer(buf_capacity, state_size, action_size, device=self._device)
        self._batch_size = batch_size

    def get_action(self, status: torch.Tensor) -> torch.Tensor:
        '''
            行動の選択
            TODO: stateが1つ，複数に限らず実行可能にする
        '''
        means, stds = self._actor_net.forward(status)
        normal = Normal(means, stds)
        actions = torch.tanh(normal.rsample()) * 2.0
        return actions
    
    def get_action_from_meanstd(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)
        
        if len(std.shape) == 1:
            std = std.unsqueeze(0)
        
        normal = Normal(mean, std)
        actions = torch.tanh(normal.rsample()) * 2.0

        return actions
    
    def update(self, observation):
        '''
            リプレイバッファの更新
            ネットワークの更新
        '''
        ### リプレイバッファの更新
        self._replayBuf.add(observation)
        if self._replayBuf.real_size() < self._batch_size:
            return 

        ### ネットワークの更新

        ## バッチの取得
        batch = self._replayBuf.get_batch(self._batch_size)
        status, rewards, actions, next_status, dones = batch

        ## Criticの更新
        next_action4update = self.get_action_from_meanstd(*self._actor_net.forward(next_status))
        q_target = rewards * self._gamma * self._critic_net_target(next_status, next_action4update) *(1 - dones)
        q_val = self._critic_net(status, actions)
        critic_loss = self._critic_loss(q_val, q_target.detach())

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        ## Actorの更新
        action4update = self.get_action_from_meanstd(*self._actor_net.forward(status))
        actor_loss = -self._critic_net.forward(status, action4update).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        ## critic_netの同期処理
        self._interval_count, do_sync = self.step_interval_count(self._interval_count)
        if do_sync:
            self._critic_net_target = deepcopy(self._critic_net)

        ## 記録
    
    def step_interval_count(self, interval_count: int) -> Tuple[int, bool]:
        interval_count = (interval_count + 1) % self._critic_sync_interval
        do_sync = (interval_count == 0)
        return interval_count, do_sync
        