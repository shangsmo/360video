from common import *


class Person:
    id = 0
    gender = "default gender"
    age = "default age"
    occupation = "default occupation"
    educationalBackground = "default educational background"
    vrExperience = "default"
    vrDevice = "default device"
    liveVrStreamingExperience = "default"

    def __init__(self, array):
        self.id = array[0]
        self.gender = array[1]
        self.age = array[2]
        self.occupation = array[3]
        self.educationalBackground = array[4]
        self.vrExperience = array[5]
        self.vrDevice = array[6]
        self.liveVrStreamingExperience = array[7]


class Video:
    experimentNo = 0
    videoNo = 0
    videoResolution = "default resolution"
    videoDuration = 0
    frameRate = 0
    frameCount = 0
    content = "default"
    category = "default category"
    bitrate = 0
    videoLink = "default link"
    dropboxLink = "default link"

    def __init__(self, experimentNo, array):
        self.experimentNo = experimentNo
        self.videoNo = array[0]
        self.videoResolution = array[1]
        self.videoDuration = array[2]
        self.frameRate = array[3]
        self.frameCount = array[4]
        self.content = array[5]
        self.category = array[6]
        self.bitrate = array[7]
        self.videoLink = array[8]
        self.dropboxLink = array[9]


class Tile:
    timeId = 0
    positionId = 0
    b2qMap = {}
    optBitrate = 0

    def __init__(self, timeId, positionId, b2qMap={}):
        self.timeId = timeId
        self.positionId = positionId
        self.b2qMap = b2qMap


class Encoder:
    hightNum = 0
    lengthNum = 0
    segmentLength = 0
    bitrateList = []

    def __init__(self, higthNum, lengthNum, segmentLength, bitrateList):
        self.hightNum = higthNum
        self.lengthNum = lengthNum
        self.segmentLength = segmentLength
        self.bitrateList = bitrateList


class Viewpoint:
    time = 0.0
    eulerangle = Eulerangle(0, 0, 0)
    quaternion = eulerangle.toQuaternion()

    def __init__(self, time, object):
        self.time = time
        if isinstance(object, Eulerangle):
            self.eulerangle = object
            self.quaternion = object.toQuaternion()
        else:
            self.quaternion = object
            self.eulerangle = object.toEulerangle()


class Chunk:
    tiles = []
    probabilitys = []

    def __init__(self, encoder, timeId=0):
        self.probabilitys = []
        self.tiles = []
        for i in range(encoder.hightNum):
            tile_row = []
            probability_row = []
            for j in range(encoder.lengthNum):
                positionId = i + j * encoder.lengthNum
                tile_row.append(Tile(timeId, positionId))
                probability_row.append(0)
            self.tiles.append(tile_row)
            self.probabilitys.append(probability_row)


class Trajectory:
    rolls = []
    pitchs = []
    yaws = []
    qxs = []
    qys = []
    qzs = []
    qws = []
    times = []
    splits = []
    interval = 0

    def __init__(self, interval, viewpoints):
        self.interval = interval
        self.rolls = []
        self.pitchs = []
        self.yaws = []
        self.qxs = []
        self.qys = []
        self.qzs = []
        self.qws = []
        self.times = []
        self.splits = []
        end = interval
        split = []
        for i in range(len(viewpoints)):
            viewpoint = viewpoints[i]
            if viewpoint.time > end:
                self.splits.append(split)
                meanViewpoint = self.mean(end, split)
                self.qxs.append(meanViewpoint.quaternion.qx)
                self.qys.append(meanViewpoint.quaternion.qy)
                self.qzs.append(meanViewpoint.quaternion.qz)
                self.qws.append(meanViewpoint.quaternion.qw)
                self.rolls.append(meanViewpoint.eulerangle.roll)
                self.pitchs.append(meanViewpoint.eulerangle.pitch)
                self.yaws.append(meanViewpoint.eulerangle.yaw)
                self.times.append(meanViewpoint.time)
                split = []
                end += interval
            split.append(viewpoint)
        self.splits.append(split)
        self.mean(end, split)
        split.clear()

    @staticmethod
    def mean(end, viewpoints):
        pitch = 0
        roll = 0
        yaw = 0
        length = len(viewpoints)
        if length == 0:
            length = 1
        for viewpoint in viewpoints:
            pitch += viewpoint.eulerangle.pitch
            roll += viewpoint.eulerangle.roll
            yaw += viewpoint.eulerangle.yaw
        pitch = pitch / length
        roll = roll / length
        yaw = yaw / length
        return Viewpoint(end, Eulerangle(yaw, pitch, roll))
