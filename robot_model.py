import numpy as np
from math import cos, sin, atan2, pi
import math

class RobotModel:
    def __init__(self, max_v, min_v, max_w, max_acc_v, max_acc_w,
                 init_x, init_y, init_yaw, init_v=0.0, init_w=0.0, robot_radius=0.3,
                 laser_min_angle=-pi/2, laser_max_angle=pi/2, laser_increment_angle=pi/6, laser_max_range=10.0):
        self.state = np.zeros(5) # state = [x,y,yaw,v,w]
        self.state[0] = init_x
        self.state[1] = init_y
        self.state[2] = init_yaw
        self.state[3] = init_v
        self.state[4] = init_w

        # physical constrain
        self.max_v = max_v
        self.min_v = min_v
        self.max_w = max_w
        self.max_acc_v = max_acc_v
        self.max_acc_w = max_acc_w
        self.robot_radius = robot_radius
        self.laser_min_angle = laser_min_angle
        self.laser_max_angle = laser_max_angle
        self.laser_increment_angle = laser_increment_angle
        self.laser_num = int(round((laser_max_angle-laser_min_angle)/laser_increment_angle)) + 1
        self.laser_max_range = laser_max_range

    def motion(self, input_u, dt):

        # constrain input velocity
        constrain = self.constrain_input_velocity(self.state, dt)
        u = np.array(input_u)
        u[0] = max(constrain[0], u[0])
        u[0] = min(constrain[1], u[0])

        u[1] = max(constrain[2], u[1])
        u[1] = min(constrain[3], u[1])

        # motion model, euler
        self.state[2] += u[1] * dt
        self.state[2] = self.normalize_angle(self.state[2])
        self.state[0] += u[0] * cos(self.state[2]) * dt
        self.state[1] += u[0] * sin(self.state[2]) * dt
        self.state[3] = u[0]
        self.state[4] = u[1]

        return self.state

    def constrain_input_velocity(self, state, dt):
        v_pre_max = min(state[3] + self.max_acc_v * dt, self.max_v)
        v_pre_min = max(state[3] - self.max_acc_v * dt, self.min_v)
        w_pre_max = min(state[4] + self.max_acc_w * dt, self.max_w)
        w_pre_min = max(state[4] - self.max_acc_w * dt, -self.max_w)

        return [v_pre_min, v_pre_max, w_pre_min, w_pre_max]

    def get_laser_scan(self, ob_list, ob_radius):
        # view ob in robot coordinate system
        new_ob_list = []
        for ob in ob_list:
            angle = atan2(ob[1]-self.state[1], ob[0]-self.state[0]) - self.state[2]
            distance = np.linalg.norm(ob-self.state[:2])
            if distance<=self.laser_max_range + ob_radius: # fetch ob in the circle
                new_ob_list.append([angle, distance])

        laser_ranges = []
        for i in range(self.laser_num):
            laser_angle = self.laser_min_angle + i * self.laser_increment_angle
            min_range = self.laser_max_range

            # calculate laser range
            for new_ob in new_ob_list:
                laser_range = self.calc_cross_point(laser_angle, new_ob, ob_radius)
                if min_range>laser_range:
                    min_range = laser_range
            laser_ranges.append(min_range)

        return laser_ranges


    def calc_cross_point(self, laser_angle, new_ob, ob_radius):
        diff_angle = new_ob[0] - laser_angle
        line_a = abs(new_ob[1] * sin(diff_angle))
        if abs(line_a) > ob_radius:
            return self.laser_max_range
        if cos(diff_angle)<=0:
            return self.laser_max_range
        line_b = abs(new_ob[1] * cos(diff_angle))
        laser_range = min(line_b - math.sqrt(ob_radius**2 - line_a**2), self.laser_max_range)

        return laser_range

    def normalize_angle(self, angle):
        norm_angle = angle % (2 * math.pi)
        if norm_angle > math.pi:
            norm_angle -= 2 * math.pi
        return norm_angle

    def set_init_state(self, init_x, init_y, init_yaw, init_v=0.0, init_w=0.0):
        self.state[0] = init_x
        self.state[1] = init_y
        self.state[2] = init_yaw
        self.state[3] = init_v
        self.state[4] = init_w

    def rot_to_angle(self, theta):
        norm_theta = self.normalize_angle((theta)-self.state[2])
        dead_zone = pi/8.0
        factor = self.max_w/dead_zone
        # angular_velocity = norm_theta * 7
        if norm_theta>dead_zone:
            angular_velocity = self.max_w
        elif norm_theta<-dead_zone:
            angular_velocity = -self.max_w
        else:
            angular_velocity = norm_theta*factor
        return angular_velocity

    # def PD_controler(self, theta):