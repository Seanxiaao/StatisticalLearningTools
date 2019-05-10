#(i, j) indicated the Q
#noisy dis indicated the O
import math, operator, re
import hmmlearn

MIN_MULTIPLIER = 0.7
MAX_MULTIPLIER = 1.3
ROUNDER = 1
INF = 999.9
PI_DIVISOR  = 1/87

def getState(Grid_world):
    A = {}
    for i in range(len(Grid_world)):
        for j in range(len(Grid_world[0])):
            if i == 0:
                if j == 0:
                    A.setdefault('(0,0)->(1,0)', 0.5)
                    A.setdefault('(0,0)->(0,1)', 0.5)
                elif j == 9:
                    A.setdefault('(0,9)->(0,8)', 0.5)
                    A.setdefault('(0,9)->(1,9)', 0.5)
                else:
                    A.setdefault('(0,{})->(0,{})'.format(j, j - 1),1/3)
                    A.setdefault('(0,{})->(0,{})'.format(j, j + 1),1/3)
                    A.setdefault('(0,{})->(1,{})'.format(j, j), 1/3)

            if i == 9:
                if j == 0:
                    A.setdefault('(9,0)->(8,0)', 0.5)
                    A.setdefault('(9,0)->(9,1)', 0.5)
                elif j == 9:
                    A.setdefault('(9,9)->(9,8)', 0.5)
                    A.setdefault('(9,9)->(8,9)', 0.5)
                else:
                    A.setdefault('(9,{})->(9,{})'.format(j, j - 1),1/3)
                    A.setdefault('(9,{})->(9,{})'.format(j, j + 1),1/3)
                    A.setdefault('(9,{})->(8,{})'.format(j, j),1/3)

            else:
                if j == 0:
                    up = Grid_world[i-1][j] == 1
                    down = Grid_world[i+1][j] == 1
                    right = Grid_world[i][j+1] == 1
                    lst = [up, down, right]
                    s = sum([1 for item in lst if item])
                    if up:
                        A.setdefault('({},{})->({},{})'.format(i,j,i-1,j), 1/s)
                    if down:
                        A.setdefault('({},{})->({},{})'.format(i,j,i+1,j), 1/s)
                    if right:
                        A.setdefault('({},{})->({},{})'.format(i,j,i,j+1), 1/s)

                if j == 9:
                    up = Grid_world[i-1][j] == 1
                    down = Grid_world[i+1][j] == 1
                    left = Grid_world[i][j-1] == 1
                    lst = [up, down, left]
                    s = sum([1 for item in lst if item])
                    if up:
                        A.setdefault('({},{})->({},{})'.format(i,j,i-1,j), 1/s)
                    if down:
                        A.setdefault('({},{})->({},{})'.format(i,j,i+1,j), 1/s)
                    if left:
                        A.setdefault('({},{})->({},{})'.format(i,j,i,j-1), 1/s)

                else:
                    up = Grid_world[i-1][j] == 1
                    down = Grid_world[i+1][j] == 1
                    right = Grid_world[i][j+1] == 1
                    left = Grid_world[i][j-1] == 1
                    lst = [up, down, right, left]
                    s = sum([1 for item in lst if item])
                    if up:
                        A.setdefault('({},{})->({},{})'.format(i,j,i-1,j), 1/s)
                    if down:
                        A.setdefault('({},{})->({},{})'.format(i,j,i+1,j), 1/s)
                    if right:
                        A.setdefault('({},{})->({},{})'.format(i,j,i,j+1), 1/s)
                    if left:
                        A.setdefault('({},{})->({},{})'.format(i,j,i,j-1), 1/s)
    return A

def getObservation(towers, Grid):
    all_prob_mat = {}
    t = 0
    for tower in towers:
        x_coor, y_coor = tower[0], tower[1]
        for i in range(len(Grid_world)):
            for j in range(len(Grid_world[0])):
                if Grid_world[i][j] == 1:
                    distance = math.sqrt((i - x_coor) ** 2 + (j - y_coor) ** 2)
                    interval = [round(MIN_MULTIPLIER * distance, ROUNDER), round(MAX_MULTIPLIER * distance, ROUNDER)]
                    pos = 1 / (((interval[1] - interval[0]) / 0.1 ** ROUNDER ) + 1 )#calculate prob

                else:
                    distance = INF
                    interval = [INF, INF]
                    pos  = 0.0
                all_prob_mat.setdefault('({},{})->{}'.format(i, j, t), [interval, pos])
        t += 1
    print(all_prob_mat)
    return all_prob_mat


def getTrace(INPUT_STATE, Observation, State, GridMatrix, towers):
    #initialization
    delta , temp = {}, 1
    for i in range(len(GridMatrix)):
        for j in range(len(GridMatrix[0])):
            for k in range(len(towers)):
                interval, multiplier = Observation['({},{})->{}'.format(i, j, k)][0], Observation['({},{})->{}'.format(i, j, k)][1]
                if  interval[0] <= INPUT_STATE[0][k] <= interval[1]:
                    temp *= multiplier # calculate b_i(O_1)
                else:
                    temp *= 0
            temp = temp * PI_DIVISOR
            delta.setdefault('({},{}),{}'.format(i, j, 0), temp)
            temp = 1

    #recursion
    for t in range(1, len(INPUT_STATE)):
        for m in range(len(GridMatrix)):
            for n in range(len(GridMatrix[0])):
                temp = 1
                for k in range(len(towers)):
                    interval, multiplier = Observation['({},{})->{}'.format(m, n, k)][0], \
                                           Observation['({},{})->{}'.format(m, n, k)][1]
                    if interval[0] <= INPUT_STATE[t][k] <= interval[1]:
                        temp *= multiplier  # calculate b_j(O_t)
                    else:
                        temp *= 0

                delta_update_temp_lst = {}
                trace_lst = {}
                for i in range(len(GridMatrix)):
                    for j in range(len(GridMatrix[0])):
                        if State.get('({},{})->({},{})'.format(i,j, m, n)):
                           delta_update_temp_lst.setdefault("({},{})".format(i, j), temp * delta['({},{}),{}'.format(i, j, t - 1)] \
                                                            * State['({},{})->({},{})'.format(i, j, m, n)])
                        else:
                           delta_update_temp_lst.setdefault("{},{}".format(i, j), 0)
                #print("delta_update_lst, ", delta_update_temp_lst)
                #max(delta_update_temp_lst.items(), key=operator.itemgetter(1))[0]
                delta.setdefault('({},{}),{}'.format(m, n, t), max(delta_update_temp_lst.values()))

    path = []
    new_dict = {}
    for key, value in delta.items():
        if key.endswith(',10'):
            new_dict.setdefault(key, delta[key])

    last_endpoint = max(new_dict, key=new_dict.get)
    path.append(last_endpoint)

    for t in range(len(INPUT_STATE) - 2, -1, -1):
        new_dict = {}
        for key,value in delta.items():
            if key.endswith(',{}'.format(t)):
                if State.get('({})'.format(re.findall('\((.*?)\)', key)[0]) + \
                      '->' + '({})'.format(re.findall('\((.*?)\)', last_endpoint)[0])
                             ):
                    new_dict.setdefault(key, delta[key] * State['({})'.format(re.findall('\((.*?)\)', key)[0]) + \
                                                            '->' + '({})'.format(re.findall('\((.*?)\)', last_endpoint)[0])])
                else:
                    new_dict.setdefault(key, 0)
        last_endpoint = max(new_dict,key=new_dict.get)
        path.append(last_endpoint)
    #for key, value in new_dict.items():
    #    print(value)
#

    return delta, path




if __name__ == '__main__':
    #initialzation
    Grid_world = []
    Grid_world.append([1 for x in range(10)])
    Grid_world.append([1 for x in range(10)])
    Grid_world.append([1,1,0,0,0,0,0,1,1,1])
    temp = [1,1,0,1,1,1,0,1,1,1]
    for x in range(4):
        Grid_world.append(temp)
    for x in range(3):
        Grid_world.append([1 for x in range(10)])

    #initialize Tower
    Tower = [[0, 0], [0, 9], [9, 0], [9, 9]]
    INPUT_STATE =  [[6.3,5.9,5.5,6.7],
                    [5.6,7.2,4.4,6.8],
                    [7.6,9.4,4.3,5.4],
                    [9.5,10.0,3.7,6.6],
                    [6.0,10.7,2.8,5.8],
                    [9.3,10.2,2.6,5.4],
                    [8.0,13.1,1.9,9.4],
                    [6.4,8.2,3.9,8.8],
                    [5.0,10.3,3.6,7.2],
                    [3.8,9.8,4.4,8.8],
                    [3.3,7.6,4.3,8.5]]

    State = getState(Grid_world)
    Observation = getObservation(Tower, Grid_world)
    d, p = getTrace(INPUT_STATE, Observation, State, Grid_world, Tower)
    print('The delta table is {}'.format(d))
    #where (i, j), t implied the state(i,j) under time t
    print('The best path is {}'.format(p[::-1]))

