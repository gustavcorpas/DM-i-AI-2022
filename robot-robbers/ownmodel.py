from cmath import log
import gym
from game.environment import RobotRobbersEnv
import numpy as np
from tf_agents import trajectories
from gym import spaces

class RobotRobbersEnvWrapper(gym.Env):
  def __init__(self):
    self.env = RobotRobbersEnv()
    self.action_space = spaces.Discrete(9)
    self.observation_space = spaces.Box(
            low=-1,
            high=128,
            shape=(53,),
            dtype=np.int32
        )

  def step(self, action):
    theaction = self._map_action(action)
    self.env.render()
    obs, reward, done, info = self.env.step(theaction)
    obs = self._map_observation(obs)
    return (obs, reward, done, info)

  def reset(self):
    obs =  self.env.reset()
    obs = self._map_observation(obs)
    return obs

  def render(self, mode='human', close=False):
    self.env.render()


  def _map_action(self, action_number):   
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

  def _map_observation(self, observation):
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
      for i in range(1):
          a = np.append(a, observation[5][i][0:1]) # CARIED CASH
      return a
