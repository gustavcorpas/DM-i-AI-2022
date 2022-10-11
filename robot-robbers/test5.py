from cmath import log
from game.environment import RobotRobbersEnv
import numpy as np
import math
env = RobotRobbersEnv()

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def perceptron(threshold, input):
    if input > threshold:
        return 1
    else:
        return 0


def scrooges(observation):
    a = [[], [], [], [], [], [], []]
    for i in range(7):
        a[i] = observation[1][i][0:2]
    return a

def scroogeLocation(scroge_x, scroge_y, p_x, p_y):
    if abs(scroge_x - p_x) <  17 and abs(scroge_y - p_y) < 17:
        return (
            True,
            scroge_x > p_x,
            scroge_y > p_y,
        )
    else:
        return (
            False,
            False,
            False,
        )

def playerLocation(index, observation):
    return (
        observation[0][index][0],
        observation[0][index][1],
    )

def coinLocation(index, observation):
    return (
        int(observation[2][index][0]),
        int(observation[2][index][1]),
    )

def dropLocation(index, observation):
    return(
        observation[3][index][0],
        observation[3][index][1],
    )

def makeMoveVert(index, dir):
    move_array[index*2] = dir

def makeMoveHor(index, dir):
    move_array[index*2 + 1] = dir

def hasCash(index, observation):
    x = int(observation[5][index][0])
    if x == 1:
        return True
    else:
        return False

# Find the point that is closest to base, 1-dimensional
def closest(base, candidate_1, candidate_2):
    
    neg_neg_1_closest = perceptron(0, ( 1 * candidate_1) + (-1 * candidate_2)              )
    neg_neg_2_closest = perceptron(0, (-1 * candidate_1) + ( 1 * candidate_2)              )

    pos_pos_1_closest = perceptron(0, (-1 * candidate_1) + ( 1 * candidate_2)              )
    pos_pos_2_closest = perceptron(0, ( 1 * candidate_1) + (-1 * candidate_2)              )

    neg_pos_1_closest = perceptron(0, ( 1 * candidate_1) + ( 1 * candidate_2) + (-2 * base))
    neg_pos_2_closest = perceptron(0, ( 1 * candidate_1) + ( 1 * candidate_2) + (-2 * base))

    pos_neg_1_closest = perceptron(0, ( 1 * candidate_1) + ( 1 * candidate_2) + (-2 * base))
    pos_neg_2_closest = perceptron(0, ( 1 * candidate_1) + ( 1 * candidate_2) + (-2 * base))

def far_away(threshold, a, b):

    far_a_b = perceptron(threshold, (-1 * a) + ( 1 * b))
    far_b_a = perceptron(threshold, ( 1 * a) + (-1 * b))

    # layer

    is_far_away = preceptron(1, (1 * far_a_b) + (1 * far_b_a))

    return is_far_away


def far_away_2d(threshold, a_x, a_y, b_x, b_y):

    far_x = far_away(threshold, a_x, b_x)
    far_y = far_away(threshold, a_y, b_y)

    # layer

    is_far_away = perceptron(1, far_x, far_y)



obs = env.reset()
observation = obs

move_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for _ in range(2000):

    move_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for index in range(5):

        player_x = observation[0][index][0] / env.width
        player_y = observation[0][index][1] / env.height
        coin_x = observation[2][index][0] / env.width
        coin_y = observation[2][index][1] / env.height
        drop_x = observation[3][round(index/2)][0] / env.width
        drop_y = observation[3][round(index/2)][1] / env.height
        has_cash = clamp(observation[5][index][0], 0, 1)

        # layer

        coin_move_right = perceptron(0, (1 * coin_x  ) + (-1 * player_x) + (-1 * has_cash))
        coin_move_left =  perceptron(0, (1 * player_x) + (-1 * coin_x  ) + (-1 * has_cash))
        coin_move_down =  perceptron(0, (1 * coin_y  ) + (-1 * player_y) + (-1 * has_cash))
        coin_move_up =    perceptron(0, (1 * player_y) + (-1 * coin_y  ) + (-1 * has_cash))

        drop_move_right = perceptron(1, (1 * drop_x  ) + (-1 * player_x) + (1  * has_cash))
        drop_move_left =  perceptron(1, (1 * player_x) + (-1 * drop_x  ) + (1  * has_cash))
        drop_move_down =  perceptron(1, (1 * drop_y  ) + (-1 * player_y) + (1  * has_cash))
        drop_move_up =    perceptron(1, (1 * player_y) + (-1 * drop_y  ) + (1  * has_cash))

        # layer

        move_right = perceptron(0, (1 * coin_move_right) + (1 * drop_move_right))
        move_left  = perceptron(0, (1 * coin_move_left ) + (1 * drop_move_left ))
        move_down  = perceptron(0, (1 * coin_move_down ) + (1 * drop_move_down ))
        move_up    = perceptron(0, (1 * coin_move_up   ) + (1 * drop_move_up   ))

        # layer

        move_array[index*2]     = move_right - move_left
        move_array[index*2 + 1] = move_down  - move_up

        #move_array = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

        #print(move_array)

    obs, reward, done, info = env.step(move_array)
    observation = obs
    env.render()
