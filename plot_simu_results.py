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


    plt.plot(p_hybird[:, 0], p_hybird[:, 1], "-r")
    plt.plot(p_dwa[:, 0], p_dwa[:, 1], "-b")
    plt.plot(p_apf[:, 0], p_apf[:, 1], "-m")

    plt.plot(e_longest[:, 0], e_longest[:, 1], "-g")

    if ob_list is not None:
        plt.plot(ob_list[:, 0], ob_list[:, 1], "ok")
    plt.axis("equal")
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.annotate('', xy=(0.4, 4.7), xytext=(-2.4, 1.9), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    plt.annotate('', xy=(12, 13.6), xytext=(12, 16), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    plt.tight_layout()

    figsize = (5, 3)
    # plot v(t)
    plt.figure(figsize=figsize)
    plt.plot(np.arange(0, len(p_hybird))*dt, p_hybird[:, 3], "-r")
    plt.plot(np.arange(0, len(p_dwa)) * dt, p_dwa[:, 3], "-b")
    plt.plot(np.arange(0, len(p_apf)) * dt, p_apf[:, 3], "-m")
    plt.plot(np.arange(0, len(e_longest)) * dt, e_longest[:, 3], "-g")
    plt.xlabel('T(s)')
    plt.ylabel('V(m/s)')
    plt.tight_layout()
    axes = plt.gca()
    xlim = axes.get_xlim()

    # plot angle(t)
    plt.figure(figsize=figsize)
    plt.plot(np.arange(0, len(p_hybird))*dt, p_hybird[:, 2], "-r")
    plt.plot(np.arange(0, len(p_dwa)) * dt, p_dwa[:, 2], "-b")
    plt.plot(np.arange(0, len(p_apf)) * dt, p_apf[:, 2], "-m")
    plt.plot(np.arange(0, len(e_longest)) * dt, e_longest[:, 2], "-g")
    plt.xlabel('T(s)')
    plt.ylabel('Angle(rad)')
    plt.tight_layout()

    # plot distance(t)
    p_hybird_distance = np.linalg.norm(p_hybird[:, :2]-e_hybird[:, :2], axis=1, keepdims=True)
    p_dwa_distance = np.linalg.norm(p_dwa[:, :2] - e_dwa[:, :2], axis=1, keepdims=True)
    p_apf_distance = np.linalg.norm(p_apf[:, :2] - e_apf[:, :2], axis=1, keepdims=True)
    plt.figure(figsize=figsize)
    plt.plot(np.arange(0, len(p_hybird))*dt, p_hybird_distance[:], "-r")
    plt.plot(np.arange(0, len(p_dwa)) * dt, p_dwa_distance[:], "-b")
    plt.plot(np.arange(0, len(p_apf)) * dt, p_apf_distance[:], "-m")
    plt.xlabel('T(s)')
    plt.ylabel('Distance(m)')
    plt.tight_layout()

    # plot strategy
    plt.figure(figsize=figsize)
    plt.plot(np.arange(0, len(p_strategy)) * dt, p_strategy[:], "-r")

    plt.xlim(xlim)
    plt.xlabel('T(s)')
    plt.ylabel('Strategy')
    plt.tight_layout()

    plt.show()