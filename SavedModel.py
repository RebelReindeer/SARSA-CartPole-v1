from ntpath import altsep
import pickle
from pole import policy, getState
import gym

filee = open(r'RL/Acrobot-v1/poleqtable.pkl', 'rb')
qvalues = pickle.load(filee)
filee.close()

env = gym.make('CartPole-v0')

for i in range(10):
    state = getState(env.reset())
    done = False

    while not done:
        env.render()
        action = policy(qvalues[state])
        observation, reward, done, info = env.step(action)
        next_state = getState(observation)
        state = next_state
        print(f'action: {action}, reward: {reward}')
    print('episode', i)

