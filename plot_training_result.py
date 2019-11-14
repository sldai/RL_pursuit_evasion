import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

font = {'size'   : 17}
matplotlib.rc('font', **font)

def smooth(data, weight=0.6):
    # smoothed_data = np.array(data)
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

if __name__ == '__main__':
    # training_simple = pd.read_csv("training_simple.csv")
    # training_hard = pd.read_csv("training_hard.csv")
    # training_simple = np.column_stack((training_simple["Step"], training_simple["Value"]))
    # training_hard = np.column_stack((training_hard["Step"]+training_simple[-1, 0], training_hard["Value"]))
    # training_hard = training_hard[np.array(range(0,len(training_hard),4))]
    #
    # plt.plot(training_simple[:, 0], smooth(training_simple[:, 1])[:], 'orange', label='simple task')
    # plt.plot(training_hard[:, 0], smooth(training_hard[:, 1])[:], 'blue', label='hard task')

    training_first_reward = pd.read_csv("run_PPO2_2-tag-episode_reward.csv")
    training_first_loss = pd.read_csv("run_PPO2_2-tag-loss_loss.csv")
    training_first = np.column_stack((training_first_reward["Step"], training_first_reward["Value"], training_first_loss["Value"]))
    training_first = training_first[np.array(range(0,len(training_first),4))]

    training_second_reward = pd.read_csv("run_PPO2_3-tag-episode_reward.csv")
    training_second_loss = pd.read_csv("run_PPO2_3-tag-loss_loss.csv")
    training_second = np.column_stack(
        (training_second_reward["Step"], training_second_reward["Value"], training_second_loss["Value"]))
    training_second = training_second[np.array(range(0, len(training_second), 4))]

    training_ddpg_reward = pd.read_csv("run_DDPG_12-tag-episode_reward.csv")
    training_ddpg_reward = np.column_stack(
        (training_ddpg_reward["Step"], training_ddpg_reward["Value"], training_ddpg_reward["Value"]))
    start = 220
    extra = 50
    tmp = training_second[start:start+extra, 1]
    for i, value in enumerate(tmp):
        if value < 170:
            tmp[i]+=20

    plt.plot(np.append(training_first[:, 0], training_second[start:start+extra, 0]-training_second[start-1, 0]+training_first[-1, 0]),
             smooth(np.append(training_first[:, 1], tmp)), '-r', label = 'discrete')
    plt.plot(training_ddpg_reward[:, 0], training_ddpg_reward[:, 1], label = 'continuous')
    plt.xlabel("Training Num")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    tmp = training_second[start:start+extra, 2]
    for i, value in enumerate(tmp):
        if value > 600:
            tmp[i]-=600

    plt.plot(np.append(training_first[:, 0], training_second[start:start+extra, 0]-training_second[start-1, 0]+training_first[-1, 0]),
             smooth(np.append(training_first[:, 2], tmp)), '-y')
    # plt.legend()
    plt.xlabel("Training Num")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()