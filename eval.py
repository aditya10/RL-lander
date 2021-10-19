import numpy as np
import torch
import gym
from torch.distributions.categorical import Categorical

def eval_model(path):
    print("Starting evaluation...")

    model = torch.load(path)
    model.eval()

    rewards = []

    for seed in range(0, 1000):
        print("\r"+str(seed+1)+"/1000", end='')
        env = gym.make('LunarLander-v2')
        env.seed(seed)

        total_reward = 0
        obv_state = torch.tensor(env.reset())
        done = False
        while not done:

            policy = model(obv_state)

            m = Categorical(policy)
            action = m.sample()

            obv_state, reward, done, _ = env.step(action.item())
            obv_state = torch.tensor(obv_state)

            total_reward += reward

        rewards.append(total_reward)

    print("\nAvg reward: "+str(np.mean(rewards)))
    print("Median reward: "+str(np.median(rewards)))
    print("Max reward: "+str(np.max(rewards)))
    successes = sum([i > 200 for i in rewards])
    print("Success rate: "+str(successes))


if __name__ == "__main__":
    eval_model('./save/GA_agent5_step_30.pt')