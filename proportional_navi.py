import math
import numpy as np

class ConfigPN():
    def __init__(self, K=5.0, dt=0.1):
        self.K = K
        self.dt = dt

def line_angle(pur_pose, tar_pose):
    return math.atan2(tar_pose[1]-pur_pose[1], tar_pose[0]-pur_pose[0])

def pn_control(config, pur_pose, tar_pose, pre_pur_pose, pre_tar_pose):
    pre_line_angle = line_angle(pre_pur_pose, pre_tar_pose)
    now_line_angle = line_angle(pur_pose, tar_pose)
    w = config.K * (now_line_angle-pre_line_angle)/config.dt
    return w
