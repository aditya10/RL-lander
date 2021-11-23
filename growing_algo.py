import numpy as np
import copy
import random
import torch
import gym
import wandb
from dynamic_nn import DynamicNeuralNetwork
from multiprocessing import Pool

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
def train_agent(agent, env, render=False, steps=30):

    best_agent = agent
    unchanged = True
    
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
            child_agent.life = 0
            best_agent = child_agent
            unchanged = False
    
    best_agent.life +=1

    return best_agent, unchanged

# This function grows the hidden size of the agent by 1, and does one round of training
def grow_agent(agent, env):
    print("Growing agent")
    new_hidden_dim = agent.hidden_dim + 1
    new_agent = DynamicNeuralNetwork(hidden_dim=new_hidden_dim)
    new_agent.init_weights()
    new_agent.running_rewards = new_agent.running_rewards[1:] + [new_agent.avg_reward]

    new_agent.load_from_agent(agent.state_dict())

    best_agent, _ = train_agent(new_agent, env)

    return best_agent

# This function grows the environment and makes it slightly more difficult
def grow_env(env):
    print("Growing environment")
    new_env = gym.make('LunarLander-v2')
    new_env.age = env.age + 1
    initial_random = env.initial_random + 10
    slope = random.choice([env.slope, -env.slope, env.slope+0.2, -env.slope-0.2])
    main_engine_power = random.choice([env.main_engine_power, env.main_engine_power+1, env.main_engine_power+3, env.main_engine_power+5])
    side_engine_power = random.choice([env.side_engine_power, env.side_engine_power+0.4, env.side_engine_power+1, env.side_engine_power+5])
    moon_friction = max(0.2, env.moon_friction - 0.05)
    new_env.set_parameters(initial_random, slope, main_engine_power, side_engine_power, moon_friction)
    return new_env


def archive(agent, env, final=False):
    # archive the agent and environment
    path = "./archive/"
    if final:
        path = "./archive/final"
    path = path+str(env.age)+"_"+str(agent.life)
    torch.save(agent, path+'.pt')
    env_params = [env.age, env.initial_random, env.slope, env.main_engine_power, env.side_engine_power, env.moon_friction]
    np.save(path+'.npy', env_params)

def stepper(pair):
    agent, env = pair

    new_population = []

    if sum(agent.running_rewards)/len(agent.running_rewards) < 200:
                
        # get the next agent
        best_agent, unchanged = train_agent(agent, env)

        if best_agent.life >= 30 and best_agent.avg_reward < 0:
            # if the agent is good enough, grow it
            best_agent = grow_agent(best_agent, env)

        # add the new agent to the population
        new_population.append((best_agent, env))

        if best_agent.avg_reward > 0:
            # if the agent is good enough, create a new environment
            new_env = grow_env(env)
            new_population.append((best_agent, new_env))
    else:
        archive(agent, env)

    return new_population

# Main loop: this is a POET-like algorithm, but with growing agent sizes!
def growing_agents():

    wandb.init(project='RL-Lander', entity='aditya10', tags=["Growing"])
    
    steps = 10000

    config = wandb.config
    config.steps = steps

    population = []
    new_population = []

    # Initialize the first population
    agent = DynamicNeuralNetwork(hidden_dim=1, hidden_layers=0)
    env = gym.make('LunarLander-v2')
    agent.init_weights()
    population.append((agent,env))

    # for each generation...
    for step in range(steps):
        #print(str(step), end = "\r")
        print(str(step)+" "+str(population[0][0].life)+" "+str(population[0][0].running_rewards), end = "\r")

        task_pool = Pool()
        results = task_pool.map(stepper, population)

        new_population = [item for sublist in results for item in sublist]

        if step % 100 == 0 and step > 0:
            # Conduct transfers
            print("Transfers TBD")
    
        # replace the old population with the new population
        population = new_population
        new_population = []      

        # log 
        info = {"step": step, "age": population[-1][1].age}
        wandb.log(info, step=step)
        
    # archive end population
    for agent, env in population:
        archive(agent, env, final=True)


if __name__ == "__main__":
    growing_agents()