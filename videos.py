import numpy as np
import torch
import gym
from torch.distributions.categorical import Categorical
import json
import time

def eval_model(path, params, name_path):
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

    for seed in range(0, 5):
        print("\r"+str(seed+1)+"/1000", end='')
        env = gym.make('LunarLander-v2')
        env = gym.wrappers.RecordVideo(env, './videos/' + str(seed)+"_"+name_path + '/')
        env.set_parameters(initial_random, slope, main_engine_power, side_engine_power, moon_friction, x_variance)
        #env.seed(seed)

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
        
        print(total_reward)
        rewards.append(total_reward)

    print("\nAvg reward: "+str(np.mean(rewards)))
    print("Median reward: "+str(np.median(rewards)))
    print("Max reward: "+str(np.max(rewards)))
    successes = sum([i > 200 for i in rewards])
    print("Success rate: "+str(successes))


if __name__ == "__main__":
    name_path = 'env4_archive/final/752_0_0'
    json_path = name_path+'.json'
    model_path = name_path+'_elite.pt'
    
    params = json.load(open(json_path))

    eval_model(model_path, params, name_path)