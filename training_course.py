import gym
from gym import spaces
from gym.envs.classic_control import rendering
from robot_model import RobotModel
from math import cos, sin, pi, atan2
import math, sympy
from dwa import dwa_control, ConfigDWA, plot_arrow
from apf import apf_control, ConfigAPF
from proportional_navi import pn_control, ConfigPN
import numpy as np
from gym.utils import seeding
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Rectangle, Polygon
import matplotlib
font = {'size'   : 14}
matplotlib.rc('font', **font)
def evader_traj(evader_init_num):
    target_init_x = None
    target_init_y = None
    target_init_yaw = None
    target_u = None
    if evader_init_num == 0:
        # evader env 1
        target_init_x = 0.0
        target_init_y = -5.0
        target_init_yaw = -pi / 2.0
        target_u = np.array([1.8, -pi / 5 * 2.0])
    elif evader_init_num == 1:
        # evader env 2
        target_init_x = 0.0
        target_init_y = 9.0
        target_init_yaw = pi - 0.3
        target_u = np.array([1.8, 0.66])
    elif evader_init_num == 2:
        # evader env 4
        target_init_x = 10.0
        target_init_y = 12.0
        target_init_yaw = -pi / 2.0
        target_u = np.array([1.8, -pi / 5])
    return target_init_x, target_init_y, target_init_yaw, target_u

def pursuit_position(pursuit_init_num):
    init_x = None
    init_y = None
    init_yaw = None
    if pursuit_init_num == 0:
        # pursuit env 0
        init_x = 2.1
        init_y = 7.5
        init_yaw = 0.0
    elif pursuit_init_num == 1:
        # pursuit env 1
        init_x = 8.0
        init_y = 6.0
        init_yaw = 0.0
    elif pursuit_init_num == 2:
        # pursuit env 2
        init_x = 12.0
        init_y = 8.0
        init_yaw = 0.0
    elif pursuit_init_num == 3:
        # pursuit env 3
        init_x = -3.0
        init_y = 0.0
        init_yaw = pi / 8.0

    return init_x, init_y, init_yaw

from PIL import Image
if __name__ == '__main__':
    # img = np.array(Image.open("pe_map.png").convert('L'), 'f')
    # img_copy = img.copy()
    # img_copy[img[:,:]>200] = 1
    # img_copy[img[:,:]<=200] = 0
    # plt.imshow(img_copy)
    # plt.show()


    max_v = 2
    min_v = 0
    max_w = pi / 3.0 * 2
    max_acc_v = 0.7
    max_acc_w = pi / 3.0 * 5
    init_x = 0.0
    init_y = 6.0
    init_yaw = pi / 8.0
    robot_radius = 0.3
    ob_radius = 0.3
    target_radius = 0.3
    dt = 0.1

    target_init_x = 10
    target_init_y = 4
    target_init_yaw = pi / 2.0
    target_u = None
    
    evader_model = RobotModel(max_v, min_v, max_w, max_acc_v, max_acc_w,
                                   target_init_x, target_init_y, target_init_yaw)

    ob_wall=[]
    top, bottom, left, right = 20.0, -10.0, -10.0, 20.0
    # top y=20
    for x in np.arange(left, right, 2*ob_radius):
        ob_wall.append(np.array([x, top]))
    # bottom y=-10
    for x in np.arange(left, right, 2*ob_radius):
        ob_wall.append(np.array([x, bottom]))
    # left
    for y in np.arange(bottom+2*ob_radius, top, 2*ob_radius):
        ob_wall.append(np.array([left, y]))
    # right
    for y in np.arange(bottom, top, 2*ob_radius):
        ob_wall.append(np.array([right, y]))
    ob_wall=np.array(ob_wall)

    ob =  np.array([[0, 2],
                                 [4.0, 2.0],
                                 [-2.0, 4.6],
                                 [5.0, 6.0],
                                 [7.0, 9.0],
                                 [12.0, 12.0],
                                 [4.0, 12.0]
                                 ])

    # ob =  np.array([[0, 2],
    #                              [4.0, 2.0],
    #                              [-2.0, 4.6],
    #                              [5.0, 6.0],
    #                              [7.0, 9.0],
    #                              [12.0, 12.0],
    #                              [4.0, 12.0],
    #                              [3.5, 15.8],
    #                              [12.1, 17.0],
    #                              [7.16, 14.6],
    #                              [8.6, 13.0],
    #                              [4.42, 10.76],
    #                 [-3.76, 8.8],
    #                 [2.0, -1.8],
    #                 [-0.16, -1.66],
    #                 [3.1, -5.1],
    #                 [0.7, 6.5],
    #                 [4.85, -3.05],
    #                 [8.0, -0.33],
    #                 [-0.3, -6.75]
    #                              ])

    # ob = np.vstack((ob, ob_wall))
    mydpi = 100
    resolution = 0.05
    width = 32.0/resolution


    # fig = plt.figure(figsize=(width/mydpi, width/mydpi), dpi=mydpi)
    fig = plt.figure()
    plt.axis("equal")
    ax = fig.add_subplot(111)

    # plot boundary
    # ax.add_patch(Rectangle(xy=(-10-1, -10-1), width=30.0+2, height=30.0+2, color="black"))
    # ax.add_patch(Rectangle(xy=(-10, -10), width=30.0, height=30.0, color="white"))

    for o in ob:
        ax.add_patch(Circle(xy=(o[0], o[1]), radius=ob_radius, color="black"))


    # ax.add_patch(Circle(xy=(-5.0, -5.0), radius=3.0, color="black"))
    # ax.add_patch(Circle(xy=(15.0, -4.0), radius=2.0, color="black"))
    for i in range(3):
        target_init_x, target_init_y, target_init_yaw, target_u = evader_traj(i)
        traj_evader = []
        evader_model.set_init_state(target_init_x, target_init_y, target_init_yaw)
        traj_evader.append(np.array(evader_model.state))
        for steps in range(120):
            evader_model.motion(target_u, dt)
            traj_evader.append(np.array(evader_model.state))
        traj_evader = np.array(traj_evader)
        plt.plot(traj_evader[:, 0], traj_evader[:, 1], "-g")
        np.save('npy_for_fig/course/simple_target_'+str(i), traj_evader)
        # np.save('npy_for_fig/course/hard_target_' + str(i), traj_evader)

    robot = []
    for i in range(4):
        init_x, init_y, init_yaw = pursuit_position(i)
        robot.append([init_x, init_y, init_yaw])
        plt.plot(init_x, init_y, "xr")
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.tight_layout()
    np.save('npy_for_fig/course/simple_ob', ob)
    # np.save('npy_for_fig/course/hard_ob', ob)
    np.save('npy_for_fig/course/simple_robot', np.array(robot))
    # np.save('npy_for_fig/course/hard_robot', np.array(robot))

    plt.annotate('', xy=(-3.3, 7.2), xytext=(-0.24, 11), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    plt.annotate('', xy=(12, 13.6), xytext=(12, 16), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    plt.annotate('', xy=(1.9, -5.27), xytext=(1.9, -3), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.xlim(-11, -11+32)
    # plt.ylim(-11, -11+32)
    # plt.axis('off')
    plt.show()