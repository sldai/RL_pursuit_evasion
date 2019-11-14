from pursuit_env_v1 import PursuitEnv1

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, MlpLstmPolicy, CnnLstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common.schedules import LinearSchedule
import gym
import numpy as np
# from evdev import InputDevice
from select import select
import threading



# class keyboard(threading.Thread):
#     foward = 103
#     left = 105
#     right = 106
#     back = 108
#     def __init__(self, threadID, name, counter, env):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.counter = counter
#         self.env = env
#
#     def run(self):
#         print("Starting " + self.name)
#         self.detectInputKey()
#         print("Exiting " + self.name)
#
#     def detectInputKey(self):
#         dev = InputDevice('/dev/input/event4')
#         # from pursuit_env_v1 import MAX_W, MIN_V, MAX_V
#         # while True:
#         #     select([dev], [], [])
#         #     for event in dev.read():
#         #         if (event.value == 1) and event.code != 0:
#         #             # print("code:%s value:%s" % (event.code, event.value))
#         #             if event.code is self.left:
#         #                 print("left")
#         #                 keyboard_u[1] = MAX_W
#         #             elif event.code is self.right:
#         #                 print("right")
#         #                 keyboard_u[1] = -MAX_W
#         #             elif event.code is self.foward:
#         #                 print("forward")
#         #                 keyboard_u[0] = MAX_V/2.0
#         #             elif event.code is self.back:
#         #                 print("back")
#         #                 keyboard_u[0] = MIN_V/2.0


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 160
    if is_solved:
        return False

if __name__ == '__main__':

    n_cpu = 1
    env = SubprocVecEnv([lambda: PursuitEnv1() for i in range(n_cpu)])  # The algorithms require a vectorized environment to run
    total_timesteps = 500000
    # schedule = LinearSchedule(total_timesteps/n_cpu, final_p=0.00001, initial_p=0.00005)
    policy_kwargs = dict(net_arch=[300, 300])
    model = PPO2.load('pursuit.pkl', env=env, tensorboard_log="./pursuit_tensorboard/")
    # model = PPO2(MlpPolicy, env, verbose=1, n_steps=128, policy_kwargs=policy_kwargs, tensorboard_log="./no_course_tensorboard/")
    # model.learn(total_timesteps=total_timesteps)
    # model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, tb_log_name="second_run")
    # model.save('pursuit_no_course')

    # model = PPO2.load("pursuit")

    obs = env.reset()
    # use for record one episode

    # dones = np.array([False])
    # while not dones[0]:
    #     env.render()
    #     action, _states = model.predict(obs,deterministic=True)
    #     action[0]=0
    #     obs, rewards, dones, info = env.step(action)

    # keyboard_u = [0, 0]
    # use for demonstration
    performance = np.zeros([20, 3])
    cnt = 0
    episode_reward = 0

    while True:

        # env.set_attr("keyboard_u", keyboard_u)

        env.render()
        action, _states = model.predict(obs,deterministic=True)
        action[0]=0
        obs, rewards, dones, info = env.step(action)
        episode_reward += rewards[0]
        if dones[0]:
            performance[cnt, 0] = episode_reward
            episode_reward = 0

            performance[cnt, 1] = env.get_attr("record_count")[0]
            # print(env.get_attr("record_count"))
            performance[cnt, 2] = env.env_method("why_done")[0]
            # print(env.env_method("why_done"))
            if int(performance[cnt, 2]) != 0:
                performance[cnt, 1] = np.inf
            cnt += 1

            break
    print(performance)
    print(np.mean(performance[:, 0]), np.min(performance[:, 1])*0.1,
          np.max(performance[:, 1])*0.1, np.mean(performance[:, 1])*0.1, len(performance[performance[:,2] == 0]),
          len(performance[performance[:, 2] == 1]), len(performance[performance[:,2] == 2]))


