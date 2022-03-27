import matplotlib.pyplot as plt
import numpy as np
import pickle
import gym

#outputs +1 - 1
#reward of +1 every timestep its not fallen
#[position of cart, velocity of cart, angle of pole, rotation rate of pole]

env = gym.make('CartPole-v1')

def explore():
    qtable = createQtable()
    for i in range(10):
        done = False
        state = getState(env.reset())
        while not done:
            env.render()
            action = policy(action_values=qtable[state])
            next_state, reward, done, info = env.step(action)
            next_state = getState(next_state)
            print('state', state, 'action', action, 'reward', reward)
            state = next_state

global cart_position_space
global cart_velocity_space
global pole_angle_space
global pole_rotation_space
global actions

actions = [0, 1]
cart_position_space = np.linspace(-4.8000002, 4.8000002, 10)
cart_velocity_space = np.linspace(-3.4028235, 3.4028235, 10)
pole_angle_space = np.linspace(-4.1887903, 4.1887903, 10)
pole_rotation_space = np.linspace(-3.4028235, 3.4028235, 10)

def getState(observation):
    cart_position, cart_velocity, pole_angle, pole_rotation = observation

    cart_pos_ind = int(np.digitize(cart_position, cart_position_space))
    cart_vel_ind = int(np.digitize(cart_velocity, cart_velocity_space))
    pole_ang_ind = int(np.digitize(pole_angle, pole_angle_space))
    pole_rot_ind = int(np.digitize(pole_rotation, pole_rotation_space))

    return (cart_pos_ind, cart_vel_ind, pole_ang_ind, pole_rot_ind)

def policy(action_values):
    return np.argmax(action_values)

def createQtable():
    qtable = dict({})
    for cart_pos in range(len(cart_position_space) +1):
        for cart_vel in range(len(cart_velocity_space) + 1):
            for pole_ang in range(len(pole_angle_space) + 1):
                for pole_rot in range(len(pole_rotation_space) + 1):
                    qtable[(cart_pos, cart_vel, pole_ang, pole_rot)] = []
    
    for state in qtable.keys():
        for action in [0, 1]:
            qtable[state].append(0)

    return qtable

def sarsa(qtable, policy, episodes, alpha, epsilon, gamma):

    returns = np.array([])

    for episode in range(episodes):
        state = getState(env.reset())
        done = False
        action = policy(state) if np.random.random() > epsilon else env.action_space.sample()
        return_ = 0
        for i in range(1000):
            observation, reward, done, info = env.step(action)
            next_state = getState(observation)
            next_action = policy(qtable[next_state])

            qsa = qtable[state][action]
            qtable[state][action] = qsa + alpha * (reward + gamma * qtable[next_state][next_action] - qsa)
            state = next_state
            action = next_action
            return_ += reward
        if episode % 1000 == 0:
            returns = np.append(returns, return_) 
            print(f'episode {episode} return {return_}')

    return qtable, returns


if __name__ == '__main__':

    p = policy
    qtable = createQtable()
    qtable, returns = sarsa(qtable, p, 500000, 0.1, 0.2, 0.99)

    model = open(r'/Users/LuisV./machinelearningpractice/RL/Acrobot-v1/poleqtable.pkl', 'wb')
    pickle.dump(qtable, model)
    model.close()

    plt.plot(returns)
    plt.show()