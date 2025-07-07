import os
import torch.optim.adam
from tqdm import tqdm
from matplotlib import pyplot as plt

# gymnasium
import gymnasium as gym
from gymnasium import Env

# torch
import torch
from torch import nn

# Actor Critic
from ActorCritic.model import ActorCriticAgent




def main():
    # モデル作成
    agent = ActorCriticAgent(0.95, 0.001, 0.1, 
                             3, 1, 
                             64, 1, "Adam", 
                             64, 3, "Adam", 200, 
                             20000, 32, torch.device("cpu"))
    
    env = gym.make("Pendulum-v1")

    
    reward_history = list()
    
    for _ in tqdm(range(500)):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action.detach().numpy()[0])
            done = truncated or terminated

            agent.update(state, action, reward, next_state, done)
            total_reward += reward

            state = next_state
        reward_history.append(total_reward)
    plt.plot(reward_history)
    plt.show()
    plt.savefig("reward_history.png")

if __name__ == '__main__':
    main()