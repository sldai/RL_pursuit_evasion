import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'size'   : 13}
matplotlib.rc('font', **font)
if __name__ == '__main__':
    dt = 0.1
    ob_list = np.array([[0, 2],
                             [4.0, 2.0],
                             [-2.0, 4.6],
                             [5.0, 6.0],
                             [7.0, 9.0],
                             [12.0, 12.0],
                             [4.0, 12.0],
                             [3.5, 15.8],
                             [12.1, 17.0],
                             [7.16, 14.6],
                             [8.6, 13.0],
                             [4.42, 10.76],
                             [-3.76, 8.8],
                             [2.0, -1.8],
                             [-0.16, -1.66],
                             [3.1, -5.1],
                             [0.7, 6.5],
                             [4.85, -3.05],
                             [8.0, -0.33],
                             [-0.3, -6.75]
                             ])
    # ob_list = np.array([[0, 2],
    #           [4.0, 2.0],
    #           [-2.0, 4.6],
    #           [5.0, 6.0],
    #           [7.0, 9.0],
    #           [12.0, 12.0],
    #           [4.0, 12.0]
    #           ])
    # ob_list = None
    prefix = "scenario3/"
    p_strategy = np.load(prefix+"p_strategy.npy")
    p_hybird = np.load(prefix+"p_hybrid.npy")
    e_hybird = np.load(prefix+"e_hybrid.npy")

    p_dwa = np.load(prefix+"p_dwa.npy")
    e_dwa = np.load(prefix+"e_dwa.npy")

    p_apf = np.load(prefix+"p_apf.npy")
    e_apf = np.load(prefix+"e_apf.npy")

    e_combine = [e_hybird, e_dwa, e_apf]
    e_length_combine = [len(e_hybird), len(e_dwa), len(e_apf)]
    e_longest = e_combine[e_length_combine.index(max(e_length_combine))]

    fig = plt.figure()
    frames = []

    fig.set_tight_layout(True)
    from matplotlib.animation import FuncAnimation
    def update(time):
        plt.clf()
        robot = plt.plot(p_apf[time, 0], p_apf[time, 1], "-xr")
        target = plt.plot(e_apf[time, 0], e_apf[time, 1], "-xb")
        if ob_list is not None:
            ob = plt.plot(ob_list[:, 0], ob_list[:, 1], "ok")
        plt.axis("equal")
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        return robot, target, ob




    ani = FuncAnimation(fig, update, frames=range(len(p_apf)), interval=10)
    ani.save('apf.gif', dpi=80, writer='imagemagick')







    # plt.show()