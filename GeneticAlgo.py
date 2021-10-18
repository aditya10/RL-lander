import numpy as np
import copy
import torch
import gym
import wandb

from network import NeuralNetwork

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

def fitness_func(agent, env, render):
    # A better fitness function
    # Evaluates an agent three times
    # proposes an updated reward and a suggested mutation power
    mutation_power = 0.01
    rewards = []

    for _ in range(3):
        reward = run_eval(agent, env, render)
        rewards.append(reward)

    avg_reward = np.mean(rewards)
    fitness_score = avg_reward

    if avg_reward > 200: # Is winning all games
        fitness_score += 200
        mutation_power = 0.0001
    elif avg_reward > 150: # Has the potiential to win a game
        fitness_score += 100
        mutation_power = 0.001
    elif avg_reward < -100: # Has performed terribly
        fitness_score -= 100
        mutation_power = 0.5
    
    return avg_reward, fitness_score, mutation_power

def getEliteIndex(population, fitness, env, eval_count=10):
    # Also pick the top 10 to run elite eval
    top10 = np.argsort(fitness)[-10:]

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

def genetic(render=False):

    # Start a new run
    wandb.init(project='RL-Lander', entity='aditya10', tags=["GA"])

    N = 1000
    T = 100
    mutation_power = 0.01
    steps = 150
    use_variable_mutation = True
    use_fitness = True

    # Save model inputs and hyperparameters
    config = wandb.config
    config.N = N
    config.T = T
    config.mutation_power = mutation_power
    config.steps = steps
    config.use_variable_mutation = use_variable_mutation
    config.use_fitness = use_fitness

    population = []
    fitness = []
    rewards = []
    mutation_powers = []

    env = gym.make('LunarLander-v2')

    # Initialize the first population
    for i in range(0, N):

        # This randomly intitilizes a neural network agent
        agent = NeuralNetwork()
        agent.init_weights()
        population.append(agent)

    # for each generation...
    for step in range(steps):
    
        # Evaluate the population
        for i in range(0, N):
            agent = population[i]

            if use_variable_mutation:
                avg_reward, fitness_score, mutation_power = fitness_func(agent, env, render)
                rewards.append(avg_reward)
                fitness.append(fitness_score)
                mutation_powers.append(mutation_power)
            elif use_fitness:
                avg_reward, fitness_score, _ = fitness_func(agent, env, render)
                rewards.append(avg_reward)
                fitness.append(fitness_score)
                mutation_powers.append(mutation_power)
            else: 
                reward = run_eval(agent, env, render)
                rewards.append(reward)
                fitness.append(reward)
                mutation_powers.append(mutation_power)

        # Next, pick the top-n agents with the highest fitness
        topT = np.argsort(fitness)[-T:]

        new_population = []
        # Create a new population:
        for _ in range(0, N-1):
            i = np.random.choice(topT) # Pick an agent at random from the topT indices
    
            # create a new agent by modifying the parameters of this agent
            parent = population[i]
            agent = copy.deepcopy(parent)

            if use_variable_mutation:
                mutation_power = mutation_powers[i]
            
            for param in agent.parameters():
                mutation = mutation_power * np.random.normal(size=tuple(param.data.shape))
                param.data += torch.DoubleTensor(mutation)
            
            new_population.append(agent)

        elite_i = getEliteIndex(population, fitness, env)
        elite_parent = population[elite_i]
        agent = copy.deepcopy(elite_parent)
        new_population.append(agent)

        # Print avg rewards for this step
        info = {"step": step, "avg_fitness": np.mean(fitness), "avg_reward": np.mean(rewards), "top_reward": np.max(rewards), "elite_fitness": fitness[elite_i], "elite_reward": rewards[elite_i]}
        print(info)
        wandb.log(info)

        del population
        population = new_population
        fitness = []
        rewards = []
        mutation_powers = []

        if step % 10 == 0:
            torch.save(agent, './save/GA_agent3_step_'+str(step)+'.pt')

    # Save best model
    best_agent = population[-1]
    torch.save(best_agent, './save/GA3_best.pt')
    
if __name__ == "__main__":
    genetic()