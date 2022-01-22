import csv
import math

import numpy as np

from prediction import *
from element import *

dataSetPath = "./vr-dataset/Formated_Data/"
experimentPath = "Experiment_"
userInfoFile = "userDemo.csv"
viewpointFile = "/video_"
dataSetType = ".csv"
videoMetaFile = "/videoMeta.csv"
userInfoList = []
videoInfoList = []


def init():
    # userInfo init
    userReader = csv.reader(open(dataSetPath + userInfoFile))
    userReader.__next__()
    for line in userReader:
        user = Person(line)
        userInfoList.append(user)
    # videoInfo init
    for experiment in ["1", "2"]:
        videoReader = csv.reader(open(dataSetPath + experimentPath + experiment + videoMetaFile))
        videoReader.__next__()
        for line in videoReader:
            video = Video(experiment, line)
            videoInfoList.append(video)


def reader(experimentNo, videoNo, usrNo):
    viewpointList = []
    viewpointReader = csv.reader(
        open(dataSetPath + experimentPath + str(experimentNo) + "/" + str(usrNo) + viewpointFile + str(
            videoNo) + dataSetType))
    viewpointReader.__next__()
    for line in viewpointReader:
        quat = Quaternion(float(line[2]), float(line[3]), float(line[4]), float(line[5]))
        viewpoint = Viewpoint(float(line[1]), quat)
        viewpointList.append(viewpoint)
    return viewpointList


def read_users(experimentNo, videoNo, users):
    all_users_trajectory = []
    # 初始化所有训练轨迹
    for user in users:
        org = reader(experimentNo, videoNo, user)
        trajectory = Trajectory(0.5, org)
        all_users_trajectory.append(trajectory)
    return all_users_trajectory


def train_nn_net():
    users = []
    for i in range(40):
        users.append(i + 1)
    all_users_trajectory = read_users(2, 8, users)
    train(2, 8, all_users_trajectory)


if __name__ == '__main__':
    init()
    users = []
    for i in range(40):
        users.append(i + 1)
    net = Net(17, 20, 8)
    net.load_state_dict(torch.load('./nnar_model/NNAR_2_3_706.pth'))
    l = Trajectory(0.5, reader(2, 3, 42))
    all_users_trajectory = read_users(2, 3, users)
    accs_yaw = []
    accs_pitch = []
    for t in range(16,len(l.yaws) - 8):
        #prediction = nnar_model([t], l, net, all_users_trajectory)
        prediction = linear_model([t, t + 1, t + 2, t + 3], [l.yaws[t:t + 4], l.pitchs[t:t + 4]], [t + 4, t + 5, t + 6, t + 7])
        #prediction = avg_model([t, t + 1, t + 2, t + 3], [l.yaws[t:t + 4], l.pitchs[t:t + 4]], [t + 4, t + 5, t + 6, t + 7])
        #prediction = ar_model([t], [l.yaws[t-16:t + 4], l.pitchs[t-16:t + 4]], [t + 4, t + 5, t + 6, t + 7])

        accura_yaw = abs(l.yaws[t + 4] - prediction[0][0]) / (2 * math.pi)
        if accura_yaw > 0.5:
            accura_yaw = 1 - accura_yaw
        accura_pitch = abs(l.pitchs[t + 4] - prediction[1][0]) / math.pi

        accs_yaw.append(accura_yaw)
        accs_pitch.append(accura_pitch)
    print(np.mean(accs_yaw))
    print(np.mean(accs_pitch))
    import matplotlib.pyplot as plt

    plt.plot(accs_yaw)
    plt.show()
    '''
    print(l.yaws[t:t+4])
    print(prediction[0])
    print(l.pitchs[t:t+4])
    print(prediction[1])
    '''
