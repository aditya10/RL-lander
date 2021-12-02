import numpy as np
import copy
import torch
import gym
import wandb
import random
import json
import time
import os
#from concurrent.futures import ProcessPoolExecutor

from dynamic_nn import DynamicNeuralNetwork

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

def fitness_func(agent, env, eval_count=10, render=False):
    # A better fitness function
    # Evaluates an agent three times
    # proposes an updated reward and a suggested mutation power

    mutation_power = 0.2
    rewards = []

    for _ in range(eval_count):
        reward = run_eval(agent, env, render)
        rewards.append(reward)

    avg_reward = np.mean(rewards)
    fitness_score = avg_reward

    if avg_reward > 200: # Is winning all games
        fitness_score += 200
        agent.mutation_power = 0.01
    elif avg_reward > 100: # Has the potiential to win a game
        fitness_score += 100
        agent.mutation_power = 0.1
    elif avg_reward < -100: # Has performed terribly
        fitness_score -= 100
        agent.mutation_power = 0.5
    else:
        agent.mutation_power = mutation_power
    
    return avg_reward, fitness_score

class MultiAgent(object):

    def __init__(self, N, T):

        self.elite_candidates = []
        self.elite_agent = DynamicNeuralNetwork(hidden_dim=1)
        self.best_agent = DynamicNeuralNetwork(hidden_dim=1)

        self.elite_reward = -10000
        self.running_avg_reward = -10000
        self.best_reward = -10000
        self.running_rewards = [-10000]*10
        self.elite_running_avg_reward = -10000

        self.elite_age = 0
        self.learning_period = 10
        
        self.N = N # Number of agents
        self.T = T # top T potential elites
    
        self.create_initial_elite_candidates()


    def create_initial_elite_candidates(self):
        # Initialize the first population
        for i in range(self.T):
            agent = DynamicNeuralNetwork(hidden_dim=1)
            agent.init_weights()
            self.elite_candidates.append(agent)

    def genetic(self, env):
        
        # Create a new population of N agents using T elites:
        agent_population = []
        agent_population.append(self.elite_agent)
        
        for _ in range(self.N-1):
            
            # create a new agent by modifying the parameters of this agent
            parent = random.choice(self.elite_candidates)
            child_agent = copy.deepcopy(parent)
            
            for param in child_agent.parameters():
                mutation = child_agent.mutation_power * np.random.normal(size=tuple(param.data.shape))
                param.data += torch.DoubleTensor(mutation)
            
            agent_population.append(child_agent)
        
        # Evaluate the population 3 times
        fitness = []
        for agent in agent_population:
            _, fitness_score = fitness_func(agent, env, eval_count=3, render=False)
            fitness.append(fitness_score)
        
        # Sort the population by fitness
        selected_population = []
        topT = np.argsort(fitness)[-self.T:]
        for i in topT:
            selected_population.append(agent_population[i])

        self.elite_candidates = selected_population
        
        # Evaluate the selected population 10 times
        rewards = []
        for agent in selected_population:
            avg_reward, fitness_score = fitness_func(agent, env, eval_count=10, render=False)
            agent.avg_reward = avg_reward
            agent.running_rewards = child_agent.running_rewards[1:] + [avg_reward]
            agent.fitness_score = fitness_score
            rewards.append(avg_reward)

        elite_agent = selected_population[np.argmax(rewards)]

        if elite_agent.avg_reward < self.elite_agent.avg_reward and self.learning_period <= 0:
            self.elite_age += 1
        
        if elite_agent.avg_reward > self.best_reward:
            self.best_agent = copy.deepcopy(elite_agent)
            self.elite_age = 0

        self.elite_agent = elite_agent
        
        self.running_avg_reward = np.mean(rewards)
        self.best_reward = self.best_agent.avg_reward
        self.elite_reward = self.elite_agent.avg_reward
        self.running_rewards = self.running_rewards[1:] + [self.elite_reward]
        self.elite_running_avg_reward = np.mean(self.running_rewards)
        self.learning_period -= 1

    def grow_agents(self):

        new_hidden_dim = self.elite_agent.hidden_dim + 1
        
        # Grow the elite_candidates
        new_elite_candidates = []
        for agent in self.elite_candidates:
            new_agent = DynamicNeuralNetwork(hidden_dim=new_hidden_dim)
            new_agent.init_weights()
            new_agent.running_rewards = agent.running_rewards
            new_agent.load_from_agent(agent.state_dict())
            new_agent.baseline_score = agent.avg_reward
            new_elite_candidates.append(new_agent)

        # Grow the elite
        new_elite = DynamicNeuralNetwork(hidden_dim=new_hidden_dim)
        new_elite.init_weights()
        new_elite.running_rewards = self.elite_agent.running_rewards
        new_elite.load_from_agent(self.elite_agent.state_dict())
        new_elite.baseline_score = self.elite_agent.avg_reward

        self.elite_agent = new_elite
        self.elite_candidates = new_elite_candidates
        self.elite_age = 0
        self.learning_period = 10

def grow_env(env):

    new_env = gym.make('LunarLander-v2')
    
    initial_random = min(env.initial_random + 200, 1500)
    slope = np.round(np.random.uniform(low=0.1, high=1.0), 1)
    main_engine_power = min(env.main_engine_power+4, np.floor(np.random.uniform(9, 40)))
    side_engine_power = np.round(np.random.uniform(0.1, 6), 1)
    moon_friction = np.round(np.random.uniform(0.1, 1.0), 1)
    x_variance = np.floor(np.random.uniform(1, 16))

    new_env.set_parameters(initial_random, slope, main_engine_power, side_engine_power, moon_friction, x_variance)

    distance = np.linalg.norm(np.array(env.get_param_array())-np.array(new_env.get_param_array()))
    return new_env, distance

def next_env(multi_agent, env):

    children = []
    rewards = []
    distances = []
    for _ in range(20):
        new_env, dist = grow_env(env)
        
        new_agent = copy.deepcopy(multi_agent)
        
        new_agent.genetic(new_env)
        
        if new_agent.elite_reward > -200 and new_agent.elite_reward < 200:
            children.append(new_env)
            rewards.append(new_agent.elite_reward)
            distances.append(dist)

    if len(children) > 0:
        best_env_i = np.argmax(distances)
        print("Found a child!")
        return children[best_env_i]
    else:
        print("Random environment...")
        random_env = grow_env(env)
        return random_env

def growing_multi_agents(config):

    print("Launching process for "+config["env"])

    wandb.init(project='RL-Lander', entity='aditya10', tags=["MultiAgent"], config=config, group="MultiAgent", name=config["env"])

    data = {"generations": []}
    with open(config["json_filepath"], 'w') as fp:
        json.dump(data, fp)
    
    if not os.path.exists(config["folderpath"]):
        os.makedirs(config["folderpath"])
        os.makedirs(config["folderpath"] + "final/")

    # Create env:
    env = gym.make('LunarLander-v2')
    env.set_param_array(config["params"])

    # Create agent manager:
    multi_agent = MultiAgent(config["N"], config["T"])

    # Train independently for config['steps']
    for step in range(config['steps']+1):

        start = time.time()
        multi_agent.genetic(env)

        # Test if solved
        if multi_agent.elite_running_avg_reward > 200:
            archive(multi_agent.best_agent, multi_agent.elite_agent, env, step, config["folderpath"], final=True)
            save_run(multi_agent, env, step, config["json_filepath"])
            print("Solved!")
            env = next_env(multi_agent, env)
            env.age += 1
            multi_agent.best_reward = -100000
            multi_agent.elite_reward = -100000
            multi_agent.running_avg_reward = -10000
            multi_agent.running_rewards = [-10000]*10
            multi_agent.elite_running_avg_reward = -10000

        # Save checkpoint
        if step % 50 == 0:
            archive(multi_agent.best_agent, multi_agent.elite_agent, env, step, config["folderpath"], final=False)

        # Save the population
        if step % 10 == 0:
            save_run(multi_agent, env, step, config["json_filepath"])

        if multi_agent.elite_age > config["max_age"]:
            multi_agent.grow_agents()
        
        end = time.time()
        logs = {"avg_reward": multi_agent.running_avg_reward, 
        "best_reward": multi_agent.best_reward, 
        "elite_reward": multi_agent.elite_reward, 
        "elite_age": multi_agent.elite_age, 
        "time": end-start, 
        "elite_hidden_size": multi_agent.elite_agent.hidden_dim, 
        "best_hidden_size": multi_agent.best_agent.hidden_dim,
        "env_age": env.age,
        "elite_avg_reward": multi_agent.elite_running_avg_reward}
        wandb.log(logs, step=step)
        print("Step: {}, Time: {}".format(step, end - start), end="\r")

def archive(agent, elite_agent, env, step, folderpath, final=False):
    # archive the agent and environment
    path = folderpath
    if final:
        path = folderpath+"final/"
    path = path+str(step)+"_"+str(env.age)+"_"+str(agent.age)
    
    torch.save(agent, path+'_best.pt')
    torch.save(elite_agent, path+'_elite.pt')
    
    env_params = env.show_param()
    with open(path+'.json', 'w') as fp:
        json.dump(env_params, fp)

def save_run(multi_agent, env, step, filename):
    update = {
        'step': step,
        'running_reward': multi_agent.running_avg_reward,
        'best_reward': multi_agent.best_reward,
        'elite_reward': multi_agent.elite_reward,
        "elite_avg_reward": multi_agent.elite_running_avg_reward,
        'elite_age': multi_agent.elite_age,
        'best_agent': multi_agent.best_agent.show_param(),
        'env': env.show_param(),
    }
    write_json(update, filename=filename)

def write_json(new_data, filename):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["generations"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)


if __name__ == "__main__":

    config1 = {
        "steps": 5000,
        "json_filepath": "env_1_results.json",
        "folderpath": "./env_1_archive/",
        "env": "env_1",
        "N": 200,
        "T": 20,
        "max_age": 10,
        "params": [0, 0, 10, 3, 1, 1]
    }
    config2 = {
        "steps": 5000,
        "json_filepath": "env_2_results.json",
        "folderpath": "./env_2_archive/",
        "env": "env_2",
        "N": 200,
        "T": 20,
        "max_age": 10,
        "params": [200, 0, 30, 1, 1, 4]
    }
    config3 = {
        "steps": 5000,
        "json_filepath": "env_3_results.json",
        "folderpath": "./env_3_archive/",
        "env": "env_3",
        "N": 200,
        "T": 20,
        "max_age": 10,
        "params": [700, 0.4, 20, 0.5, 0.7, 4]
    }
    config4 = {
        "steps": 5000,
        "json_filepath": "env_4_results.json",
        "folderpath": "./env_4_archive/",
        "env": "env_4",
        "N": 200,
        "T": 20,
        "max_age": 10,
        "params": [900, 0.5, 11, 0.3, 0.6, 8]
    }
    config5 = {
        "steps": 5000,
        "json_filepath": "env_5_results.json",
        "folderpath": "./env_5_archive/",
        "env": "env_5",
        "N": 200,
        "T": 20,
        "max_age": 10,
        "params": [1000, 0.7, 15, 0.5, 0.6, 8]
    }
    config6 = {
        "steps": 5000,
        "json_filepath": "env6_results.json",
        "folderpath": "./env6_archive/",
        "env": "env6",
        "N": 200,
        "T": 20,
        "max_age": 10,
        "params": [1500, 1, 9, 0.1, 0.2, 16]
    }
    
    growing_multi_agents(config5)
        
        
        
        



    