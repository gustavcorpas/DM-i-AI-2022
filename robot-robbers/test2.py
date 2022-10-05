from contextlib import nullcontext
from test import Agent
import numpy as np
import gym
import tensorflow as tf


# env = gym.make('CartPole-v0')

from game.environment import RobotRobbersEnv
env = RobotRobbersEnv()



# CONFIG STUFZ
input_dimensions = 57
number_of_actions = 9

lr = 0.001
n_games = 500
agent = Agent(gamma=0.99, epsilon=1.0, lr=lr,
            input_dims=input_dimensions,
            n_actions = number_of_actions, mem_size=1000000, batch_size=64,
            epsilon_end=0.01)



def map_action(action_number):   
    if action_number == 0:
        a = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # GO LEFT
    elif action_number == 1:
        a = [0, -1, 0, 0, 0, 0, 0, 0, 0, 0] # GO DOWN
    elif action_number == 2:
        a = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # GO UP
    elif action_number == 3:
        a = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # GO RIGHT
    elif action_number == 4:
        a = [1, -1, 0, 0, 0, 0, 0, 0, 0, 0] # GO UPLEFT
    elif action_number == 5:
        a = [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0] # GO DOWNLEFT
    elif action_number == 6:
        a = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] # GO UPRIGHT
    elif action_number == 7:
        a = [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0] # GO DOWNRIGHT
    elif action_number == 8:
        a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # STAY
    
    return a

def map_observation(observation):
    a = np.array([])
    for i in range(1):
        a = np.append(a, observation[0][i][0:2]) # PLAYERS
    for i in range(7):
        a = np.append(a, observation[1][i][0:2]) # ENEMIES
    for i in range(5):
        a = np.append(a, observation[2][i][0:2]) # CASH
    for i in range(3):
        a = np.append(a, observation[3][i][0:2]) # DROPSPOTS
    for i in range(5):
        a = np.append(a, observation[4][i][0:4]) # OBSTACLES
    for i in range(5):
        a = np.append(a, observation[5][i][0:1]) # CARIED CASH
    return a











# YUCK STUFZ

scores = []
eps_history = []

for i in range(n_games):
    done = False
    score = 0
    observation = map_observation(env.reset())
    while not done:
        action = agent.choose_action(observation)

        theaction = map_action(action)

        observation_, reward, done, info = env.step(theaction)
        observation_ = map_observation(observation_)

        # print('action: ', action)
        # print('observation: ', observation_)
        # print('reward: ', reward)
        # print('done: ', done)
        # print('info: ', info)

        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()
        env.render()
    eps_history.append(agent.epsilon)
    scores.append(score)

    avg_score = np.mean(scores[-100:])
    print('episode: ', i, 'score %.2f' % score,
            'avarage_score %.2f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
    
