"""
Mobile robot motion planning sample with Dynamic Window Approach
author: Atsushi Sakai (@Atsushi_twi)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from sklearn import preprocessing
sys.path.append("../../")


show_animation = True
font = {'size'   : 13}
matplotlib.rc('font', **font)

class ConfigDWA():
    # simulation parameters

    def __init__(self, max_v, min_v, max_w, max_acc_v, max_acc_w, v_reso=0.01, yawrate_reso=0.1, dt=0.1,
                 predict_time=1.0, to_goal_cost_gain=0.6, ob_gain=0.3, speed_cost_gain=0.7, robot_raduis=0.3, obstacle_radius=0.3):
        # robot parameter
        self.max_speed = max_v  # [m/s]
        self.min_speed = min_v  # [m/s]
        self.max_yawrate = max_w  # [rad/s]
        self.max_accel = max_acc_v  # [m/ss]
        self.max_dyawrate = max_acc_w  # [rad/ss]
        self.v_reso = v_reso  # [m/s]
        self.yawrate_reso = yawrate_reso  # [rad/s]
        self.dt = dt  # [s]
        self.predict_time = predict_time  # [s]
        self.to_goal_cost_gain = to_goal_cost_gain
        self.speed_cost_gain = speed_cost_gain
        self.ob_gain = ob_gain
        self.robot_radius = robot_raduis  # [m]
        self.obstacle_radius = obstacle_radius  # [m]


def motion(x, u, dt):
    # motion model

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    #  [vmin,vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def calc_trajectory(xinit, v, y, config):

    x = np.array(xinit)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt

    return traj


def calc_final_input(x, u, dw, config, goal, ob):

    xinit = x[:]
    min_cost = 10000.0
    min_u = u
    min_u[0] = 0.0
    best_traj = np.array([x])
    traj_list = []
    cost_list = []
    u_list = []
    # evalucate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_reso):
        for y in np.arange(dw[2], dw[3], config.yawrate_reso):
            traj = calc_trajectory(xinit, v, y, config)

            # calc cost
            to_goal_cost = calc_to_goal_cost(traj, goal, config)
            # speed_cost = config.speed_cost_gain * \
            #     (config.max_speed - traj[-1, 3])
            speed_cost = config.max_speed - traj[-1, 3]
            ob_cost = calc_obstacle_cost(traj, ob, config)
            # print(ob_cost)

            final_cost = to_goal_cost + speed_cost + ob_cost

            #print (final_cost)
            if not np.isinf(final_cost):
                traj_list.append(traj)
                cost_list.append([to_goal_cost, speed_cost, ob_cost])
                # print([to_goal_cost, speed_cost, ob_cost])
                u_list.append([v, y])
            # search minimum trajectory
            # if min_cost >= final_cost:
            #     min_cost = final_cost
            #     min_u = [v, y]
            #     best_traj = traj

    # normalize cost
    if len(cost_list)>0:
        cost_list = np.array(cost_list)
        cost_var = np.var(cost_list, axis=0)
        for j in range(len(cost_var)):
            if cost_var[j]!=0.0:
                cost_list[:,j] = preprocessing.minmax_scale(cost_list[:,j])

        # cost_list[:, 0] /= sum(cost_list[:, 0])
        # cost_list[:, 1] /= sum(cost_list[:, 1])
        # cost_list[:, 2] /= sum(cost_list[:, 2])
        cost_sum = cost_list[:, 0] * config.to_goal_cost_gain \
                   + cost_list[:, 1] * config.speed_cost_gain \
                   + cost_list[:, 2] * config.ob_gain
        cost_sum = cost_sum.tolist()
        best_index = cost_sum.index(min(cost_sum))
        min_u = u_list[best_index]
        best_traj = traj_list[best_index]

    return min_u, best_traj


def calc_obstacle_cost(traj, ob, config):
    # calc obstacle cost inf: collistion, 0:free
    def filter_ob(ob_list, x, active_distance):
        def is_in_active_distance(ob):
            dis = np.linalg.norm(np.array(ob) - x[:2]) - config.obstacle_radius
            return dis <= active_distance

        filtered_ob_list = list(filter(is_in_active_distance, ob_list))
        return np.array(filtered_ob_list)
    ob = filter_ob(ob.tolist(), traj[0], 5.0)
    skip_n = 1
    minr = float("inf")
    if ob is None:
        return 0
    if len(ob) is 0:
        return 0
    for ii in range(0, len(traj[:, 1]), skip_n):
        for i in range(len(ob[:, 0])):
            ox = ob[i, 0]
            oy = ob[i, 1]
            dx = traj[ii, 0] - ox
            dy = traj[ii, 1] - oy

            r = math.sqrt(dx**2 + dy**2)
            if r <= config.robot_radius + config.obstacle_radius:
                return float("Inf")  # collision

            if minr >= r:
                minr = r

    return 1.0/abs(minr-config.robot_radius - config.obstacle_radius)  # OK


def calc_to_goal_cost(traj, goal, config):
    # calc to goal cost. It is 2D norm.
    def normalize_angle(angle):
        norm_angle = angle % (2 * math.pi)
        if norm_angle > math.pi:
            norm_angle -= 2 * math.pi
        return norm_angle
    # goal_magnitude = math.sqrt(goal[0]**2 + goal[1]**2)
    # traj_magnitude = math.sqrt(traj[-1, 0]**2 + traj[-1, 1]**2)
    # dot_product = (goal[0] * traj[-1, 0]) + (goal[1] * traj[-1, 1])
    # error = dot_product / (goal_magnitude * traj_magnitude)
    # error_angle = math.acos(error)
    # cost = config.to_goal_cost_gain * error_angle
    # mid_point = int(round(config.predict_time/config.dt/2))
    mid_point = 4
    goal_line_angle = math.atan2(goal[1] - traj[mid_point, 1], goal[0]-traj[mid_point, 0])
    traj_angle = traj[mid_point, 2]


    error_angle = abs(normalize_angle(goal_line_angle-traj_angle))
    # cost = config.to_goal_cost_gain * error_angle
    cost = error_angle

    return cost


def dwa_control(x, u, config, goal, ob):
    # Dynamic Window control

    dw = calc_dynamic_window(x, config)

    u, traj = calc_final_input(x, u, dw, config, goal, ob)

    return u, traj



def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

from robot_model import RobotModel

if __name__ == '__main__':
    gx = 10.0
    gy = 12.0
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    # x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    motionModel = RobotModel(2, 0, math.pi / 3 * 2, 0.7, math.pi / 3 * 5,
                             0.0, 0.0, math.pi / 8.0, 0.0, 0.0)
    x = motionModel.state
    # goal position [x(m), y(m)]
    target_model = RobotModel(2, 0, math.pi / 3 * 2, 0.7, math.pi / 3 * 5,
                             gx, gy, -math.pi / 2.0, 0.0, 0.0)
    goal = target_model.state[:2]
    # obstacles [x(m) y(m), ....]
    ob = np.array([[-1, -1],
                   [0, 2],
                   [4.0, 2.0],
                   [5.0, 4.0],
                   [5.0, 5.0],
                   [5.0, 6.0],
                   [5.0, 9.0],
                   [8.0, 9.0],
                   [7.0, 9.0],
                   [12.0, 12.0]
                   ])

    u = np.array([0.0, 0.0])
    config = ConfigDWA(2, 0, math.pi / 3 * 2, 0.7, math.pi / 3 * 5)

    traj = np.array(x)
    traj_target = np.array(target_model.state)
    for i in range(200):
        u, ltraj = dwa_control(x, u, config, goal, ob)

        x = motionModel.motion(u, config.dt)
        target_model.motion([1.8, -math.pi/5.0], config.dt) #  [1.8, -math.pi/5.0]
        goal = target_model.state[:2]
        traj = np.vstack((traj, x))  # store state history
        traj_target = np.vstack((traj_target, target_model.state))
        # print(traj)

        if show_animation:
            plt.cla()
            plt.plot(ltraj[:, 0], ltraj[:, 1], "-c")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check goal
        if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) <= config.robot_radius*2:
            print("Goal!!")
            break

        # check collision
        collision = False
        for obstacle in ob:
            if math.sqrt((x[0] - obstacle[0]) ** 2 + (x[1] - obstacle[1]) ** 2) <= (
                        motionModel.robot_radius + config.obstacle_radius):
                collision = True
        if collision:
            print("Collision!")
            break

    print("Done")

    fig_path = 'npy_for_fig/'
    np.save(fig_path+'dwa_example_flaw/robot.npy', traj)
    np.save(fig_path+'dwa_example_flaw/target.npy', traj_target)
    np.save(fig_path+'dwa_example_flaw/ob.npy', ob)

    plt.plot(traj[:, 0], traj[:, 1], "-r")
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.grid(False)
    plt.tight_layout()


    # plot v(t)
    plt.figure(figsize=(5,2.5))
    plt.plot(np.arange(0, len(traj))*config.dt, traj[:, 3], "-b")
    plt.xlabel('T(s)')
    plt.ylabel('V(m/s)')
    plt.tight_layout()

    # plot w(t)
    plt.figure(figsize=(5,2.5))
    plt.plot(np.arange(0, len(traj))*config.dt, traj[:, 4], "-g")
    plt.xlabel('T(s)')
    plt.ylabel('W(rad/s)')
    plt.tight_layout()
    plt.show()
