import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from probability import *
from process import *


if __name__ == '__main__':
    encoder = Encoder(6, 12, 1, [])
    users = []
    for i in range(40):
        users.append(i + 1)
    net = Net(17, 20, 8)
    net.load_state_dict(torch.load('./nnar_model/NNAR_2_3_706.pth'))
    l = Trajectory(0.5, reader(2, 3, 42))
    all_users_trajectory = read_users(2, 3, users)

    tmp = []
    res_kalman = []
    res_multi = []
    res_MLE = []
    flag1 = False
    flag2 = False
    for t in range(len(l.times) - 8):
        step = 0
        prediction = nnar_model([t], l, net, all_users_trajectory)
        if flag1:
            tiles_kalman = kalman(prediction[0][step], prediction[1][step], tmp[0][step], tmp[1][step], l.splits[t + 3 + step], encoder, t + 4 + step)
            tiles_multi = multi_user(all_users_trajectory, encoder, t + 4 + step)
            if flag2:
                tiles_MLE = MLE(l.splits[t + 3 + step], encoder, t + 4 + step, tiles_kalman, tiles_multi, res_kalman[-1], res_multi[-1])
            res_kalman.append(tiles_kalman)
            res_multi.append(tiles_multi)
        flag1 = True
        flag2 = True
        tmp = prediction

