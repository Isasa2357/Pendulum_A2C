import os
import sys
import torch.optim.adam
from tqdm import tqdm
from matplotlib import pyplot as plt
import json

from typing import Dict, Union, Any

# gymnasium
import gymnasium as gym
from gymnasium import Env

# torch
import torch
from torch import nn

# Actor Critic
from ActorCritic.model import ActorCriticAgent

# usefulParam
from usefulParam import Param
from usefulParam.Param import ScalarParam

from mutil_common.mutil_common import resolve_conflict_filename, get_moveaverage

def parse_args(args: Dict[str, Any]) -> Dict[str, Any]:
    '''
    引数の解析
    '''
    for i in range(1, len(sys.argv), 2):
        # keyの妥当性確認
        if sys.argv[i] not in args.keys():
            raise RuntimeError(f'引数のkeyがvaildではない: key {sys.argv[i]}')
        
        key = sys.argv[i]
        val = sys.argv[i + 1]

        val = type(args[key])(val)
        args[key] = val
    
    # フォルダ名の調整
    args['result'] = resolve_conflict_filename(args['result'], os.path.join('project', args['project']))
    
    return args

def make_project_folder(project: str, result: str) -> None:
    # プロジェクトフォルダの作成
    project_path = os.path.join('project', project)
    os.makedirs(project_path, exist_ok=True)

    # リザルトフォルダの作成
    result_path = os.path.join(project_path, result)
    os.mkdir(result_path)

    print(f'make result folder : {result_path}')

def make_exec_cmd(args: Dict[str, Any], initial: str):
    '''
    今回と同じ条件で実行するためのコマンドを作成する．
    args: 実験条件
    initial: コマンドの先頭に置く文字列(python ~~ or uv run ~~~)
    '''
    cmd = initial
    for key, val in args.items():
        cmd += ' ' + str(key) + ' ' + str(val)
    return cmd


def make_result(args: Dict[str, Any], reward_history: list, model: ActorCriticAgent):
    '''
    結果ファイルの作成．以下を行う
    ・条件の記録
    ・報酬の推移の保存
    ・Actorのロスの推移の保存
    ・Criticのロスの推移の保存
    ・モデルの記録
    '''
    project = args['project']
    result = args['result']
    result_path = os.path.join('project', project, result)

    # 実行コマンドの作成
    py_cmd = make_exec_cmd(args, f'python {sys.argv[0]}')
    uv_cmd = make_exec_cmd(args, f'uv run {sys.argv[0]}')

    args['py_cmd'] = py_cmd
    args['uv_cmd'] = uv_cmd

    ### 条件の記録
    with open(os.path.join(result_path, 'condtion.json'), 'w') as cj:
        json.dump(args, cj, indent=4)

    ### 報酬の推移の記録
    with open(os.path.join(result_path, 'reward_history.txt'), 'w') as f:
        for reward in reward_history:
            f.write(str(reward))
            f.write('\n')
    reward_history_ma = get_moveaverage(reward_history, 100)
    plt.plot(reward_history, label='reward_history')
    plt.plot(reward_history_ma, label='ma')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig(os.path.join(result_path, 'reward_history.png'))
    plt.clf()

    ### Actorのロスの記録
    actor_loss_history = model.actor_loss_history
    actor_loss_history_ma = get_moveaverage(actor_loss_history, 100)
    with open(os.path.join(result_path, 'actor_history.txt'), 'w') as f:
        for actor_loss in actor_loss_history:
            f.write(str(actor_loss) + '\n')
    plt.plot(actor_loss_history, label='actor_loss_history')
    plt.plot(actor_loss_history_ma, label='ma')
    plt.xlabel('step')
    plt.ylabel('actor loss')
    plt.savefig(os.path.join(result_path, 'actor_loss_history.png'))
    plt.clf()

    ### Criticのロスの記録
    critic_loss_history = model.critic_loss_history
    critic_loss_history_ma = get_moveaverage(critic_loss_history, 100)
    with open(os.path.join(result_path, 'critic_loss_history.txt'), 'w') as f:
        for critic_loss in critic_loss_history:
            f.write(str(critic_loss) + '\n')
    plt.plot(critic_loss_history, label='critic_loss_history')
    plt.plot(critic_loss_history_ma, label='ma')
    plt.xlabel('step')
    plt.ylabel('critic_loss')
    plt.savefig(os.path.join(result_path, 'critic_loss_history.png'))
    plt.clf()

    ### モデルの記録
    model.save_actor('actor.pth', result_path)

def main():
    # 条件の設定
    args = {
        # Param
        'gamma': 0.95, 
        'lr': 0.005, 
        'tau': 0.1, 

        # Actor
        'Actor_hdn_chnl': 64, 
        'Actor_hdn_lays': 1, 

        # Critic
        'Critic_hdn_chnls': 64, 
        'Critic_hdn_lays': 3, 
        'Critic_optimizer': 'Adam', 
        'Critic_sync_interval': 200, 

        # RepBuf
        'buf_capacity': 20000, 
        'batch_size': 32, 

        # other
        'episodes': 500, 
        'project': 'project', 
        'result': 'result', 
        'device': 'cpu'
    }
    args = parse_args(args)

    # プロジェクトフォルダの作成
    make_project_folder(args['project'], args['result'])

    # モデル作成
    state_size = 3
    action_size = 1
    agent = ActorCriticAgent(Param.makeConstant(args['gamma']), Param.makeConstant(args['lr']), Param.makeConstant(args['tau']), 
                             state_size, action_size, 
                             args['Actor_hdn_chnl'], args['Actor_hdn_lays'], 
                             args['Critic_hdn_chnls'], args['Critic_hdn_lays'], args['Critic_optimizer'], args['Critic_sync_interval'], 
                             args['buf_capacity'], args['batch_size'], torch.device(args['device']))
    
    env = gym.make("Pendulum-v1")
    
    reward_history = list()
    
    for _ in tqdm(range(args['episodes'])):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            state = torch.tensor(state, dtype=torch.float32, device=torch.device(args['device']))
            action = agent.get_action_np(state)
            # print(action)
            # print(action.shape)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = truncated or terminated

            agent.update(state, action, reward, next_state, done)
            total_reward += reward.__float__()

            state = next_state
        reward_history.append(total_reward)
    
    make_result(args, reward_history, agent)
if __name__ == '__main__':
    main()