import csv

import numpy as np

from prediction import *

x=np.arange(0,100,1)
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
        open(dataSetPath + experimentPath + str(experimentNo) + "/" + str(usrNo) + viewpointFile + str(videoNo) + dataSetType))
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

from nntrain import Net
if __name__ == '__main__':
    init()
    net = Net(17, 20, 8)
    net.load_state_dict(torch.load('./nnar_model/NNAR_2_4_417.pth'))
    l = Trajectory(0.5, reader(2, 4, 46))
    users = []
    for i in range(40):
        users.append(i+1)
    t = 250
    nnar_model([t],l,net,read_users(2,4,users))
    print(l.yaws[t:t+4])
    print(l.pitchs[t:t+4])