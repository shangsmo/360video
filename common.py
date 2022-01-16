import numpy as np
import math


class Quaternion:
    qx = 0.0
    qy = 0.0
    qz = 0.0
    qw = 0.0

    def __init__(self, qx, qy, qz, qw):
        tmp = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        self.qx = qx/tmp
        self.qy = qy/tmp
        self.qz = qz/tmp
        self.qw = qw/tmp

    def toEulerangle(self):
        qx = self.qx
        qy = self.qy
        qz = self.qz
        qw = self.qw
        roll = math.atan2(2 * (qw * qz + qy * qx), 1 - 2 * (qx * qx + qz * qz))
        pitch = math.asin(2 * (qz * qy - qw * qx))
        yaw = - math.atan2(2 * (qw * qy + qx * qz), 1 - 2 * (qx * qx + qy * qy))
        return Eulerangle(yaw, pitch, roll)


class Eulerangle:
    yaw = 0.0
    roll = 0.0
    pitch = 0.0

    def __init__(self, yaw, pitch, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    def toQuaternion(self):
        pitch = self.pitch
        yaw = self.yaw
        roll = self.roll
        qz = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qx = - np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) - np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qy = - np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        return Quaternion(qx, qy, qz, qw)