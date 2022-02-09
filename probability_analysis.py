import matplotlib.pyplot as plt
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
    res = []
    flag = False
    for t in range(len(l.times) - 8):
        step = 0
        prediction = nnar_model([t], l, net, all_users_trajectory)
        if flag:
            tiles_kalman = kalman(prediction[0][step], prediction[1][step], tmp[0][step], tmp[1][step], l.splits[t + 3 + step], encoder, t + 4 + step)
            res.append(np.mat(tiles_kalman.probabilitys))
        flag = True
        tmp = prediction


