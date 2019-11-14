"""
Potential Field based path planner
author: Atsushi Sakai (@Atsushi_twi)
Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin, cos, atan2, asin, pi
from robot_model import RobotModel
import logging
class ConfigAPF():
    # simulation parameters

    def __init__(self, max_v, min_v, max_w, max_acc_v, max_acc_w, v_reso=0.01, yawrate_reso=0.1, dt=0.1,
                 ka=0.04, kp=0.9, p0=8.0, max_range=10.0, robot_raduis=0.3, obstacle_radius=0.3):
        # robot parameter
        self.max_speed = max_v  # [m/s]
        self.min_speed = min_v  # [m/s]
        self.max_yawrate = max_w  # [rad/s]
        self.max_accel = max_acc_v  # [m/ss]
        self.max_dyawrate = max_acc_w  # [rad/ss]
        self.v_reso = v_reso  # [m/s]
        self.yawrate_reso = yawrate_reso  # [rad/s]
        self.dt = dt  # [s]
        self.robot_radius = robot_raduis  # [m]
        self.obstacle_radius = obstacle_radius  # [m]

        # potential parameter

        self.ka = ka  # attractive potential gain
        self.kp = kp  # repulsive potential gain
        self.p0 = p0  # potential area width [m]
        self.max_range = max_range


def calc_u(config, x, v_tar, theta_tar, theta_robo_tar, p_robo_tar, v_ob, theta_ob, theta_robo_ob, p_robo_ob):
    p_robo_tar = min(p_robo_tar, config.max_range)
    ob_num = len(v_ob)
    n = np.zeros(ob_num)
    beta = np.zeros(ob_num)
    for i in range(ob_num):
        Pi = (p_robo_ob[i]) - config.obstacle_radius
        n[i] = config.kp * Pi**(-1) \
               / (p_robo_ob[i]) * (Pi**-1 - config.p0**-1)
        beta[i] = (n[i] * (p_robo_ob[i])) / (config.ka * (p_robo_tar))
    logging.info("n = %s", n)
    logging.info("beta = %s", beta)
    psi_hat_y = sin(theta_robo_tar)
    for i in range(ob_num):
        psi_hat_y -= beta[i] * sin(theta_robo_ob[i])

    psi_hat_x = cos(theta_robo_tar)
    for i in range(ob_num):
        psi_hat_x -= beta[i] * cos(theta_robo_ob[i])
    logging.info("theta_robo_ob = %s", str(theta_robo_ob))
    logging.info("psi = %s", str(theta_robo_tar))
    logging.info("tan(psi_hat) = %s", str([psi_hat_y, psi_hat_x]))
    psi_hat = atan2(psi_hat_y, psi_hat_x)
    logging.info("psi_hat = %s", str(psi_hat))

    term1 = (v_tar)*cos(theta_tar-theta_robo_tar) + config.ka * (p_robo_tar)
    logging.info("term1 = %s * cos(%s - %s) + %s * %s = %s", v_tar, theta_tar, theta_robo_tar, config.ka, p_robo_tar, term1)
    for i in range(ob_num):
        term1 -= beta[i] * (v_ob[i]) * cos(theta_ob[i] - theta_robo_ob[i])

    logging.info("term1 = %s", str(term1))

    v = math.sqrt(term1**2 + ((v_tar)*sin(theta_tar - psi_hat))**2)
    theta_term_tmp = (v_tar) * sin(theta_tar-psi_hat) / v
    logging.info("theta_term_tmp = %s * sin(%s - %s)/%s = %s", v_tar, theta_tar, psi_hat, v, theta_term_tmp)
    if abs(theta_term_tmp)>1.0:
        theta_term_tmp = theta_term_tmp/abs(theta_term_tmp)
    theta = psi_hat + asin(theta_term_tmp)

    logging.info("[v, theta] = %s", str([v, theta]))
    return [v, theta]


def apf_control(config, x_robot, x_tar, ob_list):
    filtered_ob_list = filter_ob(config, ob_list, x_robot, config.p0)
    logging.info(x_robot)
    logging.info(filtered_ob_list)
    v_tar = x_tar[3]
    theta_tar = x_tar[2]
    theta_robo_tar = atan2(x_tar[1]-x_robot[1], x_tar[0]-x_robot[0])
    p_robo_tar = np.linalg.norm(x_tar[:2]-x_robot[:2])
    ob_num = len(filtered_ob_list)
    v_ob = np.zeros(ob_num)
    theta_ob = np.zeros(ob_num)
    theta_robo_ob = []
    p_robo_ob = []
    for i in range(ob_num):
        tmp = np.array(filtered_ob_list[i]) - x_robot[:2]
        theta_robo_ob.append(atan2(tmp[1], tmp[0]))
        p_robo_ob.append(np.linalg.norm(tmp))
    theta_ob = np.array(theta_ob)
    p_robo_ob = np.array(p_robo_ob)
    u = calc_u(config, x_robot, v_tar, theta_tar, theta_robo_tar, p_robo_tar, v_ob, theta_ob, theta_robo_ob, p_robo_ob)
    return u

def filter_ob(config, ob_list, x, active_distance):
    def is_in_active_distance(ob):
        dis = np.linalg.norm(np.array(ob) - x[:2]) - config.obstacle_radius
        return dis<=active_distance
    filtered_ob_list = list(filter(is_in_active_distance, ob_list))
    return filtered_ob_list



def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)





def main(gx=8, gy=2):
    show_animation = True
    print(__file__ + " start!!")
    max_v = 2
    min_v = 0
    max_w = pi / 3 * 2
    max_acc_v = 0.7
    max_acc_w = pi / 3.0 * 5
    init_x = 0.0
    init_y = 6
    init_yaw = pi / 8.0
    robot_radius = 0.3
    ob_radius = 0.3
    goal_radius = 0.3

    pursuitor_model = RobotModel(max_v, min_v, max_w, max_acc_v, max_acc_w,
                                      init_x, init_y, init_yaw)

    target_model = RobotModel(max_v, min_v, max_w, max_acc_v, max_acc_w,
                              gx, gy, -pi/2.0)

    # obstacles [x(m) y(m), ....]
    ob = np.array([
                   [0, 2],
                   [4.0, 2.0],
                   [5.0, 4.0],
                    [5.0, 6.0],

                   [7.0, 9.0]
                   ])


    config = ConfigAPF(2, 0, math.pi/3, 0.7, math.pi/3)

    traj = np.array(pursuitor_model.state)
    traj_u = np.array([0, 0])

    traj_target = np.array(target_model.state)
    for i in range(2000):
        logging.info("")
        logging.info(i)
        u = apf_control(config, pursuitor_model.state, target_model.state, ob)
        traj_u = np.vstack((traj_u, u))
        print(u)
        u[1] = pursuitor_model.rot_to_angle(u[1])
        # u[0]=0
        x = pursuitor_model.motion(u, config.dt)
        traj = np.vstack((traj, x))  # store state history

        goal = target_model.motion([1.8, pi/5], config.dt)
        traj_target = np.vstack((traj_target, goal))
        # print(traj)

        if show_animation:
            plt.cla()

            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check goal
        if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) <= (pursuitor_model.robot_radius + target_model.robot_radius):
            print("Goal!!")
            break

        # check collision
        collision = False
        for obstacle in ob:
            if math.sqrt((x[0] - obstacle[0]) ** 2 + (x[1] - obstacle[1]) ** 2) <= (
                pursuitor_model.robot_radius + config.obstacle_radius):
                collision = True
        if collision:
            print("Collision!")
            break

    prefix = "apf_flaw/4_"
    np.save(prefix + "traj.npy", traj)
    np.save(prefix + "traj_target.npy", traj_target)
    np.save(prefix + "traj_u.npy", traj_u)
    print("Done")
    if show_animation:
        plt.plot(traj[:, 0], traj[:, 1], "-r")
        plt.plot(traj_target[:, 0], traj_target[:, 1], "-g")


    fig = plt.gcf()
    # fig.set_size_inches(6,6)
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    # plt.axis("auto")

    plt.tight_layout()
    print(plt.xlim((-1.488585497312541, 14.14373130420207)))
    print(plt.ylim((0.4404926013282211, 12.089089959876205)))

    plt.figure()
    plt.plot(np.arange(0, len(traj))*config.dt, traj_u[:, 1], label=r'Input $\theta$')
    plt.plot(np.arange(0, len(traj))*config.dt, traj[:, 2], label=r'Actual $\theta$')
    plt.xlabel('T(s)')
    plt.ylabel(r'$\theta$(rad)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%H:%M:%S', filename="test.log", filemode='w')

    main()
