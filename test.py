# import gym
# env = gym.make('Reacher-v2')
#
# env.reset()
# while True:
#     env.render()

from dwa import dwa_control, ConfigDWA, plot_arrow
import math
import numpy as np
from robot_model import RobotModel
from matplotlib import pyplot as plt

show_animation = True
if __name__ == '__main__':
    gx = 10
    gy = 10
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    # x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    motionModel = RobotModel(2, 0, math.pi / 3 * 2, 0.7, math.pi / 3 * 5,
                             0.0, 0.0, math.pi / 8.0, 0.0, 0.0)
    x = motionModel.state
    # goal position [x(m), y(m)]
    target_model = RobotModel(2, 0, math.pi / 3 * 2, 0.7, math.pi / 3 * 5,
                             gx, gy, math.pi / 2.0, 0.0, 0.0)
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

    for i in range(1000):
        u, ltraj = dwa_control(x, u, config, goal, ob)

        x = motionModel.motion(u, config.dt)
        target_model.motion([1.0, 0], config.dt)
        goal = target_model.state[:2]
        traj = np.vstack((traj, x))  # store state history

        # print(traj)

        if show_animation:
            plt.cla()
            plt.plot(ltraj[:, 0], ltraj[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check goal
        if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")

    plt.plot(traj[:, 0], traj[:, 1], "-r")
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')


    plt.figure()
    # plot v(t)
    plt.figure(figsize=(5,2))
    plt.plot(np.arange(0, len(traj)*config.dt, config.dt), traj[:, 3], "-b")
    plt.xlabel('T(s)')
    plt.ylabel('V(m/s)')

    # plot w(t)
    plt.figure(figsize=(5,2))
    plt.plot(np.arange(0, len(traj)*config.dt, config.dt), traj[:, 4], "-g")
    plt.xlabel('T(s)')
    plt.ylabel('W(rad/s)')
    plt.show()
