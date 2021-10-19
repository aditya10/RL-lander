from numpy import double, mod
import numpy as np
import torch
import gym
from torch.distributions.categorical import Categorical


def render_model(examples=10):

    env = gym.make('LunarLander-v2')

    model = torch.load('./save/model2.pt')
    model.eval()

    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.data)
    #     mutation = 0.5*np.random.normal(size=tuple(param.data.shape))
    #     #mutation = 10
    #     param.data += torch.DoubleTensor(mutation)
    #     print(param.data)

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
    #trainLander(render=False)
    render_model()