import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin, cos, atan2, asin, pi
from robot_model import RobotModel
import logging
import matplotlib
def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)
    return None

def normalize_angle(angle):
    norm_angle = angle % (2 * math.pi)
    if norm_angle > math.pi:
        norm_angle -= 2 * math.pi
    return norm_angle

font = {'size'   : 14}
matplotlib.rc('font', **font)

if __name__ == '__main__':
    dt = 0.1


    prefix = "apf_flaw/static_"
    apf_static_traj = np.load(prefix + "traj.npy")
    apf_static_traj_target = np.load(prefix + "traj_target.npy")
    apf_static_traj_u = np.load(prefix + "traj_u.npy")
    for i in range(len(apf_static_traj_u)):
        apf_static_traj_u[i, 1] = normalize_angle(apf_static_traj_u[i, 1])
    ob = np.array([
                   [0, 2],
                   [4.0, 2.0],
                   [5.0, 4.0],
                                [5.0, 6.0],
                   [7.0, 9.0]
                   ])
    np.save('apf_flaw/ob_3',ob)
    plt.figure(figsize=None)
    plt.plot(apf_static_traj[:, 0], apf_static_traj[:, 1], "-r")
    plt.plot(apf_static_traj_target[:, 0], apf_static_traj_target[:, 1], "-g")
    # plt.plot(apf_static_traj[int(len(apf_static_traj)/2):int(len(apf_static_traj)/2)+1, 0], apf_static_traj[int(len(apf_static_traj)/2):int(len(apf_static_traj)/2)+1, 1], "->r")
    # plt.plot(apf_static_traj_target[int(len(apf_static_traj_target) / 2):int(len(apf_static_traj_target) / 2)+1, 0], apf_static_traj_target[int(len(apf_static_traj_target) / 2):int(len(apf_static_traj_target) / 2)+1, 1], "->g")
    plot_arrow(apf_static_traj[-1, 0], apf_static_traj[-1, 1], apf_static_traj[-1, 2])
    plt.plot(apf_static_traj[-1, 0], apf_static_traj[-1, 1], "xr")
    plt.plot(apf_static_traj_target[-1, 0], apf_static_traj_target[-1, 1], "xb")
    plt.plot(ob[:, 0], ob[:, 1], "ok")
    # plt.annotate('', xy=(12, 4.8), xytext=(12, 2.7), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')

    plt.tight_layout()
    # plt.axis("equal")
    plt.xlim((-1.488585497312541, 14.14373130420207))
    plt.ylim((0.4404926013282211, 12.089089959876205))

    plt.figure(figsize=(6,2.5))
    plt.plot(np.arange(0, len(apf_static_traj_u))*dt, apf_static_traj_u[:, 1], label=r'Input $\theta$')
    plt.plot(np.arange(0, len(apf_static_traj))*dt, apf_static_traj[:, 2], label=r'Actual $\theta$')
    plt.xlabel('T(s)')
    plt.ylabel(r'$\theta$(rad)')
    # plt.xlim(-1, len(apf_static_traj)*dt+5)
    plt.ylim(-4, 4)
    # plt.legend()
    plt.tight_layout()

    plt.show()

