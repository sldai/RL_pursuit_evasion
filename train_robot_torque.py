import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from pursuit_env_torque import PursuitEnvTorque

if __name__ == '__main__':

    env = PursuitEnvTorque()
    env = DummyVecEnv([lambda: env])

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    # param_noise = AdaptiveParamNoiseSpec
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    # action_noise = None
    policy_kwargs = dict(layers=[300, 300])
    # model = DDPG(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, param_noise=param_noise, action_noise=action_noise, tensorboard_log="./pursuit_torque_board/", actor_lr=0.00005, critic_lr=0.00005)
    model = DDPG.load('ddpg_mountain', env=env,tensorboard_log="./pursuit_torque_board/", critic_lr=0.0001)
    model.learn(total_timesteps=400000)
    model.save("ddpg_mountain")

    del model # remove to demonstrate saving and loading

    model = DDPG.load("ddpg_mountain")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()