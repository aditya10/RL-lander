import numpy as np
import gym

def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the slope
                  s[6] is the angular speed
                  s[7] 1 if first leg has contact, else 0
                  s[8] 1 if second leg has contact, else 0
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[6]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[7] or s[8]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False, prints=True):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False:
                break

        if prints and (steps % 20 == 0 or done):
            print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if done:
            break
    if render:
        env.close()
    return total_reward


if __name__ == "__main__":
    params = {
        'initial_random': [0, 100, 400, 800, 1000, 1200, 1500, 1700, 2000],
        'slope': [0, 0.2, -0.4, 0.5, -0.7, 0.9, -1.0, 1.2],
        'main_engine_power': [0.1, 0.6, 1, 5, 10, 15, 20, 25, 30],
        'side_engine_power': [0.1, 0.5, 1, 5, 10, 15, 20, 25, 30],
        'moon_friction': [1, 0.8, 0.5, 0.2],
        'x_variance': [0.1, 1, 2, 4, 8, 16]
    }


    
    env1 = [0, 0, 15, 3, 1, 1]
    env2 = [200, 0, 60, 1, 1, 4]
    env3 = [700, 0.4, 20, 0.5, 0.7, 4]
    env4 = [900, 0.5, 11, 0.3, 0.6, 8]
    env5 = [1000, 0.7, 10, 0.5, 0.6, 8]
    env6 = [1500, 1, 9, 0.1, 0.2, 16]
    envs = [env1, env2, env3, env4, env5, env6]
    #envs = [env1]
    
    for env_params in envs:
        rewards = []
        env = gym.make('LunarLander-v2')
        env.set_parameters(initial_random=env_params[0], slope=env_params[1], main_engine_power=env_params[2], side_engine_power=env_params[3], moon_friction=env_params[4], x_variance=env_params[5])
        print(env.show_param())
        for i in range(100):
            r = demo_heuristic_lander(env, render=False, prints=False)
            rewards.append(r)
            print(i, end='\r')
        print(np.mean(rewards))
        print(np.max(rewards))
        print(len([i for i in rewards if i > 200]))