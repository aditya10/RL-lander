import numpy as np
import copy
import torch
import json
import gym
import wandb

from LunarLander import LunarLander
from network import NeuralNetwork

# 1. Start a new run
wandb.init(project='RL-Lander', entity='aditya10', tags=["GA"])

def run_eval(agent, env, render):

    action_space = np.array([0,1,2,3])

    total_reward = 0

    # run this agent thorough the envionment
    obv_state = torch.tensor(env.reset())
    done = False
    while not done:

        if render:
            still_open = env.render()
            if still_open == False:
                break

        policy = agent(obv_state)
        policy_np = policy.detach().numpy()
        action = np.random.choice(action_space, p=policy_np)
        obv_state, reward, done, _ = env.step(action)
        obv_state = torch.tensor(obv_state)
        total_reward += reward
    
    return total_reward

def genetic(render=False):

    N = 1000
    T = 100
    mutation_power = 0.01
    steps = 10000

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.N = N
    config.T = T
    config.mutation_power = mutation_power
    config.steps = steps

    population = []
    rewards = []

    env = gym.make('LunarLander-v2')

    # Initialize the first population
    for i in range(0, N):

        # This randomly intitilizes a neural network agent
        agent = NeuralNetwork()
        population.append(agent)

    # for each generation...
    for step in range(steps):
    
        # Evaluate the population
        for i in range(0, N):
            agent = population[i]
            total_reward = run_eval(agent, env, render)
            rewards.append(total_reward)

        # Next, pick the top-n agents with the highest rewards
        topT = np.argsort(rewards)[-T:]

        new_population = []
        # Create a new population:
        for _ in range(0, N-1):
            i = np.random.choice(topT) # Pick an agent at random from the topT indices
    
            # create a new agent by modifying the parameters of this agent
            parent = population[i]
            agent = copy.deepcopy(parent)

            for param in agent.parameters():
                mutation = mutation_power * np.random.normal(size=1)[0]
                param.data += mutation
            
            new_population.append(agent)

        elite_i = getEliteIndex(population, rewards, env)
        elite_parent = population[elite_i]
        agent = copy.deepcopy(elite_parent)
        new_population.append(agent)

        # Print avg rewards for this step
        # print("\r", end='')
        info = {"step": step, "avg_reward": np.mean(rewards), "top_reward": np.max(rewards)}
        print(info)

        wandb.log(info)

        del population
        population = new_population
        rewards = []

        if step % 30 == 0:
            with open("GA_run.txt", "a") as file_object:
                file_object.write(json.dumps(info)+"\n")
            torch.save(agent, './save/GA_agent2.pt')

    # Save model
    best_agent = population[-1]
    torch.save(best_agent, './save/GA_best_agent2.pt')
    
def getEliteIndex(population, rewards, env, eval_count=10):
    # Also pick the top 10 to run elite eval
    top10 = np.argsort(rewards)[-10:]

    sum_rewards = [0]*len(population)
    
    for i in top10:
        agent = population[i]

        total_reward = 0
        for _ in range(eval_count): 
            total_reward += run_eval(agent, env, False)

        sum_rewards[i] = total_reward
    
    # pick the one with max reward:
    elite_i = np.argmax(sum_rewards)
    
    return elite_i

if __name__ == "__main__":
    genetic()