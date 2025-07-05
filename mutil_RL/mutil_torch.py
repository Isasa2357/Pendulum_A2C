'''
    torch自作便利関数
'''

from typing import List

import torch
from torch import nn

############################## 文字列 → nnモジュール 変換 ##############################

def conv_str2ActivationFunc(module_str: str) -> nn.Module:
    '''
        文字列から活性化関数のnn.Moduleへ変換する
    '''
    mapping = {
        'ReLU': nn.ReLU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'Softplus': nn.Softplus(),
        'Identity': nn.Identity()
    }
    if module_str in mapping:
        return mapping[module_str]
    else:
        raise ValueError(f"Unknown activation function: {module_str}")

def factory_LinearReLU_ModuleList(in_chnls: int, hdn_chnls: int, hdn_layers: int, out_chnls: int ,out_act: str="") -> nn.ModuleList:
    '''
        LinearとReLUが積み重なったネットワークを作成
        最終層はカスタム可能

        Args:
            in_chnls: 入力チャネル
            hdn_chnls: 隠れ層のチャネル数
            hdn_lays: 隠れ層の層数
            out_chnls: 出力層のチャネル数
            out_module: 最終層の指定
    '''
    lays = nn.ModuleList()

    # 入力層
    lays.append(nn.Linear(in_chnls, hdn_chnls))
    lays.append(nn.ReLU())


    # 隠れ層
    for _ in range(hdn_layers):
        lays.append(nn.Linear(hdn_chnls, hdn_chnls))
        lays.append(nn.ReLU())

    # 出力層
    lays.append(nn.Linear(hdn_chnls, out_chnls))
    if out_act != "":
        lays.append(conv_str2ActivationFunc(out_act))

    return lays

def factory_LinearReLU_Sequential(in_chnls: int, hdn_chnls: int, hdn_layers: int, out_chnls: int ,out_act: str="") -> nn.Sequential:
    '''
        LinearとReLUが積み重なったネットワークを作成
        最終層はカスタム可能

        Args:
            in_chnls: 入力チャネル
            hdn_chnls: 隠れ層のチャネル数
            hdn_lays: 隠れ層の層数
            out_chnls: 出力層のチャネル数
            out_module: 最終層の指定
    '''
    sequential = nn.Sequential(*factory_LinearReLU_ModuleList(in_chnls, hdn_chnls, hdn_layers, out_chnls, out_act))
    return sequential