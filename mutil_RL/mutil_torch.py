'''
    torch自作便利関数
'''

import torch
from torch import nn

############################## 文字列 → nnモジュール 変換 ##############################

def conv_str2ActivationFunc(module_str: str) -> nn.Module:
    '''
        文字列から活性化関数のnn.Moduleへ変換する
    '''

    if module_str == 'ReLU':
        return nn.ReLU()
    
    return nn.ReLU()

def factory_LinearReLU_ModuleList(in_chnls: int, hdn_chnls: int, hdn_layers: int, out_chnls: int ,out_module: str=""):
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
    if out_module != "":
        lays.append(conv_str2ActivationFunc(out_module))

    return lays