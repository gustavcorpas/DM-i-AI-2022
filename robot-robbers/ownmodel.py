from game.environment import RobotRobbersEnv
import random

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

env = RobotRobbersEnv()

env.max_n_scrooges = 0
env.max_n_cashbags = 5
env.max_n_dropspots = 3

state = env.reset()


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(6, 10, 4)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(env.n_robbers * 2)
])
moves = [random.randint(-1,1) for _ in range(env.n_robbers * 2)]
state, reward, is_done, info = env.step(moves)
prediction = model([state]);
print(_)


i = 0

while i < 5:
  i += 1
  moves = [random.randint(-1,1) for _ in range(env.n_robbers * 2)]
  state, reward, is_done, info = env.step(moves)
  print(moves)
  print(state)
  print(reward)
  print(is_done)
  print(info)
