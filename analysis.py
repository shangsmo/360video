import math

import numpy as np

from process import *
import matplotlib.pyplot as plt
import numpy
import matplotlib.ticker as mticker

experimentNo = 1
videoNo = 0
usrNo = 4
interval = 1

if __name__ == '__main__':
    viewpointList = reader(experimentNo, videoNo, usrNo)
    trajectory = Trajectory(interval, viewpointList)
    v_pitch = []
    v_yaw = []
    x_roll = []
    x_pitch = []
    x_yaw = []
    v_roll = []
    for i in range(len(trajectory.yaws)-1):
        v_yaw.append(abs(trajectory.yaws[i+1]-trajectory.yaws[i]))
        v_roll.append(abs(trajectory.rolls[i + 1] - trajectory.rolls[i]))
        v_pitch.append(abs(trajectory.pitchs[i + 1] - trajectory.pitchs[i]))
        x_yaw.append(abs(trajectory.yaws[i]))
        x_pitch.append(abs(trajectory.pitchs[i]))
        x_roll.append(abs(trajectory.rolls[i]))
    print(np.mean(x_yaw))
    print(np.mean(x_pitch))
    print(np.mean(x_roll))
    v_yaw = numpy.sort(x_yaw)
    v_roll = numpy.sort(x_roll)
    v_pitch = numpy.sort(x_pitch)

    p = 1. * numpy.arange(len(v_pitch)) / float(len(v_pitch) - 1)
    fig = plt.figure()
    fig.suptitle(r'CDFs')
    ax2 = fig.add_subplot(111)
    roll = ax2.plot(v_roll, p, label=r'$\psi$')
    pitch = ax2.plot(v_pitch, p, label=r'$\theta$')
    yaw = ax2.plot(v_yaw, p, label=r'$\phi$')
    ax2.set_xlabel(r"rad/s")
    plt.xscale('log')
    ax2.legend()

    plt.show()