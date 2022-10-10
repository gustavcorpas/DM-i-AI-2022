import numpy as np
from fastapi import APIRouter
from models.dtos import RobotRobbersPredictResponseDto, RobotRobbersPredictRequestDto


router = APIRouter()


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


move_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]





@router.post('/predict', response_model=RobotRobbersPredictResponseDto)
def predict(request: RobotRobbersPredictRequestDto):
   
    obs = request.state
    for index in range(5):


        p_x, p_y = playerLocation(index, obs)

        smalldist = 1000
        small_i = 0
        for i in range(5):
            tx, ty = coinLocation(i, obs)
            scroges = scrooges(obs)
            for scrog in scroges:
                care, scrog_ahead, scrog_below = scroogeLocation(scrog[0], scrog[1], tx, ty)
                if not care:
                    dist = abs(tx-p_x) + abs(ty - p_y)
                    if dist < smalldist:
                        smalldist = dist
                        small_i = i
            target_x, target_y = coinLocation(small_i, obs)

        if hasCash(index, obs):
            smalldist = 1000
            small_i = 0
            for i in range(3):
                tx, ty = dropLocation(i, obs)
                scroges = scrooges(obs)
                for scrog in scroges:
                    care, scrog_ahead, scrog_below = scroogeLocation(scrog[0], scrog[1], tx, ty)
                    if not care:
                        dist = abs(tx-p_x) + abs(ty - p_y)
                        if dist < smalldist:
                            smalldist = dist
                            small_i = i
            target_x, target_y = dropLocation(small_i, obs)
        
        diff = target_x - p_x
        if diff > 1:
            diff = 1
        elif diff < -1:
            diff = -1

        makeMoveVert(index, diff)
        
        diff = target_y - p_y
        if diff > 1:
            diff = 1
        elif diff < -1:
            diff = -1

        makeMoveHor(index, diff)

        if hasCash(index, obs):
            scroges = scrooges(obs)
            for scrog in scroges:
                care, scrog_ahead, scrog_below = scroogeLocation(scrog[0], scrog[1], p_x, p_y)
                if care:
                    if scrog_ahead:
                        makeMoveVert(index, -1)
                    else:
                        makeMoveVert(index, 1)
                    if scrog_below:
                        makeMoveHor(index, -1)
                    else:
                        makeMoveHor(index, 1)
   

    moves = move_array
   

    return RobotRobbersPredictResponseDto(
        moves=moves
    )
