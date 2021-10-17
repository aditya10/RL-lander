import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from LunarLander import LunarLander

import wandb

from network import NeuralNetwork

# 1. Start a new run
wandb.init(project='RL-Lander', entity='aditya10')

eps = np.finfo(np.float32).eps.item()


def calc_discount_rewards(rewards, gamma=0.99):
    G=0
    returns = []
    for r in rewards[::-1]:
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.min()) / (returns.max() - returns.min())
    return returns

def calc_fixed_rewards(rewards):
    returns = np.ones_like(rewards)
    if np.sum(rewards) > 0:
        print("win")
        return torch.tensor(100*returns)
    else:
        return torch.tensor(-1*returns)

def trainLander(render=False):
    
    lr = 0.001
    num_episodes = 5000
    gamma = 0.98


    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.learning_rate = lr
    config.num_episodes = num_episodes
    config.gamma = gamma

    env = LunarLander()
    action_space = np.array([0,1,2,3])

    model = NeuralNetwork()
    # 3. Log gradients and model parameters
    wandb.watch(model)

    optimizer = Adam(model.parameters(), lr=lr)

    # an episode is a collection of states, actions, and the total reward collected

    # Every episode goes like this:
    # 1. start with new env
    # 2. pass the start state into the NN to generate a prob distribution for actions 1...4
    # 3. sample an action from this prob distr to get a new state
    # 4. repeat until done
    # At the end of the episode, update the weights of the model
    # Option 1: if won, then set set all rewards to +1, else -1
    # Option 2: use discounted rewards
    
    seeds = [1, 2, 3, 4, 5, 6, 245, 34 ,34246, 6543]

    for e in range(0, num_episodes):
        #seed = int(np.random.choice(seeds))
        #print(seed)
        #env.seed(seed)
        rewards = []
        log_policies = [] # log of the probabilities of policy given by the network
        states = []

        obv_state = torch.tensor(env.reset())
        states.append(obv_state)
        done = False
        while not done:
            
            if render:
                still_open = env.render()
                if still_open == False:
                    break

            policy = model(obv_state)
            #policy_np = policy.detach().numpy() # TODO: modify this for GPU
            #action = np.random.choice(action_space, p=policy_np)
            #log_prob = torch.log(policy)
            m = Categorical(policy)
            action = m.sample()
            log_policies.append(m.log_prob(action))

            obv_state, reward, done, _ = env.step(action.item())
            obv_state = torch.tensor(obv_state)
            
            states.append(obv_state)
            rewards.append(reward)

        # Now that we have an episode, we can calculate the update

        optimizer.zero_grad()

        returns = calc_discount_rewards(rewards, gamma)
        #returns = calc_fixed_rewards(rewards)

        objectives = []
        for i, r in enumerate(returns):
            # If the log_policy is less negative, and the action follows the policy, then 
            objective = -log_policies[i] * r
            objectives.append(objective)
        
        loss = torch.stack(objectives).sum()

        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        # 4. Log metrics to visualize performance
        wandb.log({"episode": e, "loss": loss, "reward": np.sum(rewards), "returns": torch.sum(returns).item()})
        print({"episode": e, "loss": loss, "reward": np.sum(rewards), "returns": torch.sum(returns).item()})

    # Save model
    torch.save(model, './save/model.pt')


def render_model(examples=10):

    env = LunarLander()

    model = torch.load('./save/model.pt')
    model.eval()

    for e in range(0, examples):

        total_reward = 0

        obv_state = torch.tensor(env.reset())

        done = False
        while not done:
            
            still_open = env.render()
            if still_open == False:
                break

            policy = model(obv_state)

            m = Categorical(policy)
            action = m.sample()

            obv_state, reward, done, _ = env.step(action.item())
            obv_state = torch.tensor(obv_state)

            total_reward += reward

        print(total_reward)



if __name__ == "__main__":
    trainLander(render=False)
    #render_model()