import numpy as np
import copy
import torch
import gym
from dynamic_nn import DynamicNeuralNetwork
import json
from multiprocessing import Pool
import time

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

# This functtion optimizes the agent over the environment
# The agent is the elite agent from the last generation
# Create a population of 10 mutations of this agent, then evaluate them and select a new elite
# If the old elite is better than the new elite, keep the old elite, set unchanged=True
def train_agent(agent, env, render=False, steps=10):

    best_agent = agent
    
    # Create a new population of 30 agents:
    for _ in range(0, steps):

        # create a new agent by modifying the parameters of this agent
        child_agent = copy.deepcopy(agent)
        
        for param in child_agent.parameters():
            mutation = child_agent.mutation_power * np.random.normal(size=tuple(param.data.shape))
            param.data += torch.DoubleTensor(mutation)
        
        # Evaluate the population 10 times
        avg_reward, fitness_score = fitness_func(child_agent, env, eval_count=10, render=render)

        # If the new agent is better than the old elite, replace the old elite
        if fitness_score > best_agent.fitness_score:
            child_agent.avg_reward = avg_reward
            child_agent.running_rewards = child_agent.running_rewards[1:] + [avg_reward]
            child_agent.fitness_score = fitness_score
            child_agent.age = 0
            best_agent = child_agent
    
    best_agent.age +=1

    return best_agent


# Aims to find the next top 3 agents by using GA over a population of 30 agents extracted from the previous top 3
def genetic(agent, env):
    
    # Create a new population of 30 agents:
    agent_population = []
    agent_population.append(agent)

    for _ in range(5):
        child_agent = copy.deepcopy(agent)
        child_agent.init_random_weights()
        agent_population.append(child_agent)
    
    for _ in range(24):
        
        # create a new agent by modifying the parameters of this agent
        #parent = random.choice(agents)
        child_agent = copy.deepcopy(agent)
        
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
    topT = np.argsort(fitness)[-6:]
    for i in topT:
        selected_population.append(agent_population[i])
    
    # Evaluate the selected population 10 times
    fitness = []
    for agent in selected_population:
        avg_reward, fitness_score = fitness_func(agent, env, eval_count=10, render=False)
        agent.avg_reward = avg_reward
        agent.running_rewards = child_agent.running_rewards[1:] + [avg_reward]
        agent.fitness_score = fitness_score
        fitness.append(fitness_score)

    elite_agent = selected_population[np.argmax(fitness)]

    if elite_agent.avg_reward > agent.best_reward:
        elite_agent.best_reward = elite_agent.avg_reward

    # Only reset age if it is making progress
    running_avg_reward = sum(agent.running_rewards)/len(agent.running_rewards)
    if (elite_agent.avg_reward > running_avg_reward+10) and elite_agent.avg_reward > agent.best_reward:
        elite_agent.age = 0
    else:
        elite_agent.age += 1
    
    return elite_agent

# This function grows the hidden size of the agent by 1, and does one round of training
def grow_agent(agent, env):
    print("\nGrowing agent")
    new_hidden_dim = agent.hidden_dim + 1
    new_agent = DynamicNeuralNetwork(hidden_dim=new_hidden_dim)
    new_agent.init_weights()
    
    new_agent.running_rewards = agent.running_rewards
    new_agent.load_from_agent(agent.state_dict())
    new_agent.baseline_score = agent.avg_reward
    best_agent = genetic(new_agent, env)

    return best_agent

def archive(agent, env, step, final=False):
    # archive the agent and environment
    path = "./base_archive/"
    if final:
        path = "./base_archive/final/"
    path = path+str(step)+"_"+str(env.age)+"_"+str(agent.age)
    
    torch.save(agent, path+'.pt')
    
    env_params = env.show_param()
    with open(path+'.json', 'w') as fp:
        json.dump(env_params, fp)

def stepper(pair):
    agent, env = pair
                
    # get the next agent
    best_agent = genetic(agent, env)

    if (best_agent.age >= 50 and best_agent.avg_reward > best_agent.baseline_score) or best_agent.age >= 100:
        # if the agent's capacity is saturated, grow it
        best_agent = grow_agent(best_agent, env)

    new_pair = (best_agent, env)

    return new_pair

def growing_agents(config, envs):

    population = []
    new_population = []

    # Initialize results file
    data = {"generations": []}
    with open('results.json', 'w') as fp:
        json.dump(data, fp)

    # Initialize the population
    # Create an agent-environent pair for every env
    print("Initializing population...")
    for age, env_params in enumerate(envs):
        agent = DynamicNeuralNetwork(hidden_dim=1)
        agent.init_weights()

        env = gym.make('LunarLander-v2')
        env.set_parameters(initial_random=env_params[0], slope=env_params[1], main_engine_power=env_params[2], side_engine_power=env_params[3], moon_friction=env_params[4], x_variance=env_params[5])
        env.set_age(age)
        
        print(env.show_param())

        population.append((agent,env))

    # Train independently for config['steps']
    for step in range(config['steps']+1):

        start = time.time()

        results = []
        # if config['multi']:
        #     task_pool = Pool()
        #     results = task_pool.map(stepper, population)
        #     task_pool.close()
        # else:
        for pair in population:
            results.append(stepper(pair))

        for pair in results:

            agent, env = pair

            # Test if solved
            running_avg_reward = sum(agent.running_rewards)/len(agent.running_rewards)
            if running_avg_reward > 200:
                archive(agent, env, step, final=True)
            else:
                new_population.append(pair)

            if step % 100 == 0:
                archive(agent, env, step, final=False)
                
        population = new_population
        new_population = []

        # Save the population
        if step % 10 == 0:
            save_run(population, step)
        
        end = time.time()
        print("Step: {}, Time: {}".format(step, end - start), end="\r")


def save_run(population, step):
    population_updates = []
    for pair in population:
        agent, env = pair
        update = {
            'agent': agent.show_param(),
            'env': env.show_param(),
        }
        population_updates.append(update)
    new_data = {
        'step': step,
        'population_size': len(population),
        'population': population_updates,
    }
    write_json(new_data)

# function to add to JSON
def write_json(new_data, filename='results.json'):
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

    config = {
        "steps": 5000
    }

    # params = {
    #     'initial_random': [0, 100, 400, 800, 1000, 1200, 1500, 1700, 2000],
    #     'slope': [0, 0.2, -0.4, 0.5, -0.7, 0.9, -1.0, 1.2],
    #     'main_engine_power': [0.1, 0.6, 1, 5, 10, 15, 20, 25, 30],
    #     'side_engine_power': [0.1, 0.5, 1, 5, 10, 15, 20, 25, 30],
    #     'moon_friction': [1, 0.8, 0.5, 0.2],
    #     'x_variance': [0.1, 1, 2, 4, 8, 16]
    # }

    env1 = [0, 0, 15, 3, 1, 1]
    env2 = [200, 0, 60, 1, 1, 4]
    env3 = [700, 0.4, 20, 0.5, 0.7, 4]
    env4 = [900, 0.5, 11, 0.3, 0.6, 8]
    env5 = [1000, 0.7, 10, 0.5, 0.6, 8]
    env6 = [1500, 1, 9, 0.1, 0.2, 16]

    envs = [env1, env2, env3, env4, env5, env6]
    #envs = [env1]

    growing_agents(config, envs)