import torch
from torch import nn
from torch import optim
from torch.distributions import Normal

from typing import List, Tuple, Type, Callable

from ReplayBuffer.Buffer import ReplayBuffer

from usefulParam.ScalarParam import *

class ContinuousActorNet(nn.Module):
    def __init__(self, in_chnls, hdn_chnls, hdn_lays, out_chnls):
        self._in_chnls = in_chnls
        self._hdn_chnls = self._hdn_chnls
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
        self._std_layer = nn.Linear(self._hdn_lays, self._out_chnls)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self._laysList:
            x = layer(x)
        mean = self._mean_layer(x)
        std = self._std_layer(x)
        return mean, std

class CriticNet(nn.Module):
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
        self._laysList.append(nn.Linear(self._hdn_chnls, self._out_chnls))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self._laysList:
            x = layer(x)
        return x

class ActorCritic:
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
        self._critic_net = CriticNet(state_size, critic_hdn_chnls, critic_hdn_lays, action_size)
        self._critic_net_target = CriticNet(state_size, critic_hdn_chnls, critic_hdn_lays, action_size)
        self._critic_optimizer = critic_optimizer(self._critic_net.parameters(), lr=self._lr)
        self._critic_sync_interval = critic_sync_interval

        # リプレイバッファ
        self._replayBuf = ReplayBuffer(buf_capacity, state_size, action_size, device=self._device)
        self._batch_size = batch_size

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        '''
            行動の選択
            TODO: stateが1つ，複数に限らず実行可能にする
        '''
        mean, std = self._actor_net.forward(state)
        normal = Normal(mean, std)
        action = torch.tanh(normal.rsample()) * 2.0
        return action
    
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

        ## 推論


        ## 損失の計算


        ## ネットワークの更新


        ## critic_netの同期処理


        ## 記録