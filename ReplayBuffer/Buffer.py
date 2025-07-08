from ReplayBuffer.SamplingTree import SamplingTree
from ReplayBuffer.ScalarParam import *

import torch
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, 
                 state_size: int, action_size: int, reward_size: int=1, done_size: int=1, 
                 state_type: torch.dtype=torch.float32, action_type: torch.dtype=torch.float32, reward_type: torch.dtype=torch.float32, done_type: torch.dtype=torch.int8, 
                 device: torch.device=torch.device("cpu")):
        self._capacity = capacity
        self._device = device

        # サイズ
        self._state_size = state_size
        self._action_size = action_size
        self._reward_size = reward_size
        self._done_size = done_size
        
        self._write_idx = 0     # バッファへの書き込み位置
        self._real_size = 0     # バッファへ格納された量．バッファが最大までたまると，capacityと等しくなる

        # バッファ
        self._status = torch.empty(capacity, self._state_size, dtype=state_type, device=self._device)
        self._actions = torch.empty(capacity, self._action_size, dtype=action_type, device=self._device)
        self._rewards = torch.empty(capacity, self._reward_size, dtype=reward_type, device=self._device)
        self._next_status = torch.empty(capacity, self._state_size, dtype=state_type, device=self._device)
        self._dones = torch.zeros(capacity, self._done_size, dtype=done_type, device=self._device)

    def get_batch(self, batch_size: int) -> list[torch.Tensor]:
        '''
        バッチの取り出し
        Args:
            batch_size: 取り出すバッチサイズ
        Ret:
            取り出したバッチ(Batch)
        '''
        indics = random.sample(range(self._real_size), batch_size)

        extract_status = self._status[indics]
        extract_actions = self._actions[indics]
        extract_rewards = self._rewards[indics]
        extract_next_status = self._next_status[indics]
        extract_dones = self._dones[indics]

        batch = [extract_status, extract_actions, extract_rewards, extract_next_status, extract_dones]

        return batch

    def add(self, state, action, reward, next_state, done) -> None:
        '''
        バッファへ要素を追加する

        Args:
            observation: バッファへ加える要素
            ovservation = [state, action, reward, next_state, done]
        '''

        self._status[self._write_idx] = torch.tensor(state, dtype=self._status.dtype, device=self._device)
        self._actions[self._write_idx] = torch.tensor(action, dtype=self._actions.dtype, device=self._device)
        self._rewards[self._write_idx] = torch.tensor(reward, dtype=self._rewards.dtype, device=self._device)
        self._next_status[self._write_idx] = torch.tensor(next_state, dtype=self._next_status.dtype, device=self._device)
        self._dones[self._write_idx] = torch.tensor(done, dtype=self._dones.dtype, device=self._device)
        
        self._write_idx = (self._write_idx + 1) % self._capacity
        self._real_size = min(self._real_size + 1, self._capacity)

    def reset(self):
        '''
        バッファの内容を全て初期化
        '''
        self = ReplayBuffer(self._capacity, 
                            self._state_size, self._action_size, self._reward_size, self._done_size, 
                            self._status.dtype, self._actions.dtype, self._rewards.dtype, self._dones.dtype, self._device)

    def real_size(self) -> int:
        '''
        バッファに格納された要素数
        '''
        return self._real_size

    def capacity(self) -> int:
        '''
        バッファに格納可能な限界数
        '''
        return len(self._status)
    
    def __len__(self):
        return self._real_size
    
    def to(self, device: torch.device):
        '''
            デバイスの変更
        '''
        self._status.to(device)
        self._actions.to(device)
        self._rewards.to(device)
        self._next_status.to(device)
        self._dones.to(device)
    
    def adjust_squeeze(self, data: torch.Tensor):
        if len(data.shape) == 0:
            data = data.unsqueeze(0)
        return data


class PrioritizedExperienceReplayBuffer:
    def __init__(self, capacity: int, state_size: list, action_size: list, alpha: ScalarParam, beta: ScalarParam, device: str):
        self._capacity = capacity
        self._priorities = SamplingTree(capacity)

        self._write_idx = 0
        # self._observations = [None] * self._capacity

        self._status = torch.empty(capacity, state_size, dtype=torch.float32, device=device)
        self._actions = torch.empty(capacity, action_size, dtype=torch.int32, device=device)
        self._rewards = torch.empty(capacity, dtype=torch.float32, device=device)
        self._next_status = torch.empty(capacity, state_size, dtype=torch.float32, device=device)
        self._dones = torch.zeros(capacity, dtype=torch.int32, device=device)

        self._alpha = alpha
        self._beta = beta

        self._device = device

    # パラメータの更新
    def step_alpha(self):
        self._alpha.step()

    def step_beta(self):
        self._beta.step()
    
    def step(self):
        self.step_alpha()
        self.step_beta()
    
    # 重み
    def calc_weights(self, priorities: list) -> list:
        priority_total = self._priorities.total()
        select_probs = [priority / priority_total for priority in priorities]
        weights = [(self._priorities.real_size() * select_prob)**self._beta.value() for select_prob in select_probs]
        max_weight = max(weights)
        return [weight / max_weight for weight in weights]

    # 優先度
    def calc_priorites(self, loss) -> torch.Tensor:
        priorities = [((l + 1e-6)**self._alpha.value()).tolist() for l in loss]
        return priorities
    
    def update_priorities(self, td_diffs: list, indics: list):
        # print(f'td diffs type: {td_diffs}')
        # print(f'indics type: {type(indics)}')
        new_priorities = self.calc_priorites(td_diffs)
        # rint(f'new priorities type: {type(new_priorities)}')
        self._priorities.update(indics, new_priorities)

    # バッファの更新
    def add(self, new_observation):
        # 優先度更新
        write_index = 0
        if self._priorities.real_size() == 0:
            write_index = self._priorities.add(1.0)
        else:
            write_index = self._priorities.add(self._priorities.max_leaf())
        
        # バッファ更新
        state, action, reward, next_state, done = new_observation
        self._status[write_index] = torch.tensor(state, device=self._device)
        self._actions [write_index] = torch.tensor(action, device=self._device)
        self._rewards[write_index] = torch.tensor(reward)
        self._next_status[write_index] = torch.tensor(next_state, device=self._device)
        self._dones[write_index] = torch.tensor(done, device=self._device)

    # バッチの取り出し
    def get_batch(self, batch_size: int) -> list[list, list, list]:
        priorities, indics = self._priorities.get_samples(batch_size)
        weights = self.calc_weights(priorities)
        
        extract_status = self._status[indics]
        extract_actions = self._actions[indics]
        extract_rewards = self._rewards[indics]
        extract_next_status = self._next_status[indics]
        extract_dones = self._dones[indics]

        observations = [extract_status, extract_actions, extract_rewards, extract_next_status, extract_dones]

        return observations, priorities, indics, weights
    
    def __len__(self):
        return self._priorities.real_size()