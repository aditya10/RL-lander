import numpy as np
import torch
import gym
from torch.distributions.categorical import Categorical
import json

def eval_model(path, params):
    print("Starting evaluation...")

    model = torch.load(path)
    model.eval()

    rewards = []

    initial_random = params['initial_random']
    slope = params['slope']
    main_engine_power = params['main_engine_power']
    side_engine_power = params['side_engine_power']
    moon_friction = params['moon_friction']
    x_variance = params['x_variance']

    print(params)
    print(model.show_param())

    for seed in range(0, 1000):
        print("\r"+str(seed+1)+"/1000", end='')
        env = gym.make('LunarLander-v2')
        env.set_parameters(initial_random, slope, main_engine_power, side_engine_power, moon_friction, x_variance)
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
    path = 'env2_archive/final/337_20_0'
    json_path = path+'.json'
    model_path = path+'_best.pt'

    params = json.load(open(json_path))

    eval_model(model_path, params)