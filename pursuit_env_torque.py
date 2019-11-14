import logging

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

logger = logging.getLogger(__name__)

MAX_V = 2.0
MIN_V = 0.0
MAX_W = pi / 3.0 * 2


class PursuitEnvTorque(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.max_v = MAX_V
        self.min_v = MIN_V
        self.max_w = MAX_W
        self.max_acc_v = 0.7
        self.max_acc_w = pi / 3.0 * 5
        self.init_x = 0.0
        self.init_y = 6.0
        self.init_yaw = pi / 8.0
        self.robot_radius = 0.3
        self.ob_radius = 0.3
        self.target_radius = 0.3
        self.dt = 0.1

        self.target_init_x = 10
        self.target_init_y = 4
        self.target_init_yaw = pi / 2.0
        self.target_u = None

        self.pursuitor_model = RobotModel(self.max_v, self.min_v, self.max_w, self.max_acc_v, self.max_acc_w,
                                          self.init_x, self.init_y, self.init_yaw)

        self.evader_model = RobotModel(self.max_v, self.min_v, self.max_w, self.max_acc_v, self.max_acc_w,
                                       self.target_init_x, self.target_init_y, self.target_init_yaw)

        self.config_dwa = ConfigDWA(self.max_v, self.min_v, self.max_w, self.max_acc_v, self.max_acc_w,
                                    robot_raduis=self.robot_radius, obstacle_radius=self.ob_radius, dt=self.dt)

        self.confg_apf = ConfigAPF(self.max_v, self.min_v, self.max_w, self.max_acc_v, self.max_acc_w,
                                   obstacle_radius=self.ob_radius)

        self.confg_pn = ConfigPN()
        """env parameters need reset

        """
        self.ob_list = None

        # self.ob_list = None


        self.last_obs = None
        self.last_pur_state = None
        self.last_tar_state = None
        # trajectory
        self.traj = np.array(self.get_state())

        # state done
        self.collision = False
        self.catch = False

        # actions

        # self.v_reso = 0.1
        # self.w_reso = pi/18.0
        #
        #
        # self.action_table = []
        # for v in np.arange(self.min_v, self.max_v+self.v_reso, self.v_reso):
        #     for w in np.arange(-self.max_w, self.max_w+self.w_reso, self.w_reso):
        #         self.action_table.append(np.array([v, w]))

        # self.action_num = len(self.action_table)
        self.action_num = 2
        self.action_space = spaces.Box(np.array([-1]), np.array([+1])) # v accelerate, w accelerate

        # observations = laser, goal_angle, goal_distance
        self.percept_region = np.array([-4, 4, -2, 6])  # left, right, behind, ahead
        self.basic_sample_reso = 0.1  # sampling resolution of percept region
        self.sample_reso_scale = 1.5  # sampling density increases with distance from

        def cal_samples(sample_length, basic_sample_reso, sample_reso_scale):
            # get sample number
            # x = sympy.Symbol('x')
            # result_dict = sympy.solve(
            #     [basic_sample_reso * (1 - sample_reso_scale ** x) / (1 - sample_reso_scale)
            #      - abs(sample_length)], [x])
            # print(result_dict)
            a = math.log(sample_length / basic_sample_reso * (sample_reso_scale - 1) + 1)
            b = math.log(sample_reso_scale)
            # print(str(a)+", "+str(b))
            x = a / b
            return int(x)

        left_samples = right_samples = cal_samples(abs(self.percept_region[0]), self.basic_sample_reso,
                                                   self.sample_reso_scale)
        behind_samples = cal_samples(abs(self.percept_region[2]), self.basic_sample_reso, self.sample_reso_scale)
        ahead_samples = cal_samples(abs(self.percept_region[3]), self.basic_sample_reso, self.sample_reso_scale)

        self.sample_map_center = np.array([left_samples - 1, behind_samples - 1])
        self.sample_map_width = left_samples + right_samples - 1
        self.sample_map_height = behind_samples + ahead_samples - 1
        self.percept_sample_map = []

        for i in range(self.sample_map_width):
            tmp_list_y = []
            for j in range(self.sample_map_height):
                x = self.basic_sample_reso * (1 - self.sample_reso_scale ** abs(i - self.sample_map_center[0])) \
                    / (1 - self.sample_reso_scale)
                x = x if i - self.sample_map_center[0] >= 0 else -x
                y = self.basic_sample_reso * (1 - self.sample_reso_scale ** abs(j - self.sample_map_center[1])) \
                    / (1 - self.sample_reso_scale)
                y = y if j - self.sample_map_center[1] >= 0 else -y
                tmp_list_y.append(np.array([x, y]))
            self.percept_sample_map.append(tmp_list_y)

        # self.obs_num = self.pursuitor_model.laser_num + 2
        # high = np.hstack((np.zeros(self.pursuitor_model.laser_num)+self.pursuitor_model.laser_max_range, pi, 50))
        # low = np.hstack((np.zeros(self.pursuitor_model.laser_num), -pi, 0))
        # self.observation_space = spaces.Box(low, high, dtype=np.float)

        self.obs_num = 2 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_num,), dtype=float)
        # seed
        self.np_random = None
        self.seed()

        # time limit
        self.limit_step = 300
        self.count = 0

        self.action_plot = 0

        # record
        self.pursuit_strategy = None
        self.traj_pursuitor = None
        self.traj_evader = None

        # intelligent evasion mode
        self.evasion_mode = False
        self.config_evasion = ConfigDWA(self.max_v * 0.9, self.min_v, self.max_w, self.max_acc_v, self.max_acc_w,
                                        robot_raduis=self.robot_radius, obstacle_radius=self.ob_radius, dt=self.dt)
        self.evasion_route = None
        self.evasion_route_index = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        observation, reward, done, info = None, None, None, None
        total_reward = 0
        for _ in range(random.randint(3, 3)):
            observation, reward, done, info = self.mini_step(action)
            total_reward += reward
            self.count += 1
            if done:
                self.record_count = self.count
                prefix = "scenario3/"
                # np.save(prefix+"p_strategy.npy", np.array(self.pursuit_strategy))
                # np.save(prefix+"p_hybrid.npy", np.array(self.traj_pursuitor))
                # np.save(prefix+"e_hybrid.npy", np.array(self.traj_evader))
                #
                # np.save(prefix+"p_dwa.npy", np.array(self.traj_pursuitor))
                # np.save(prefix+"e_dwa.npy", np.array(self.traj_evader))
                #
                # np.save(prefix+"p_apf.npy", np.array(self.traj_pursuitor))
                # np.save(prefix+"e_apf.npy", np.array(self.traj_evader))
                break

        return observation, total_reward, done, info

    def mini_step(self, action):
        self.action_plot = action
        self._set_action(action)

        observation = self._get_obs()
        done = self._is_done(observation)
        if self.count >= self.limit_step:
            done = True
        reward = self._compute_reward(observation)
        self.last_obs = observation

        return observation, reward, done, {}

    def reset(self):
        self.init_terrain(2)
        self.init_env_from_list(random.randint(0, 0))
        if self.evasion_mode:
            self.init_evader_route(random.randint(0, 2))
        else:
            self.init_evader_circle(random.randint(2, 2))
        # self.init_env_from_list(3, 1)
        self.pursuitor_model.set_init_state(self.init_x, self.init_y, self.init_yaw)

        self.evader_model.set_init_state(self.target_init_x, self.target_init_y, self.target_init_yaw)

        self.count = 0

        # trajectory
        self.traj = np.array(self.get_state())

        obs = self._get_obs()
        self.last_obs = obs
        self.last_pur_state = self.get_state()
        self.last_tar_state = self.get_goal()

        # record
        self.pursuit_strategy = []
        self.traj_pursuitor = []
        self.traj_pursuitor.append(self.get_state())
        self.traj_evader = []
        self.traj_evader.append(self.get_goal())

        # intelligent evasion
        self.keyboard_u = [0.0, 0.0]
        return obs

    def close(self):
        pass

    def render(self, mode='human'):
        plt.cla()
        x = self.get_state()
        goal = self.get_goal()
        ob = self.ob_list

        plt.plot(x[0], x[1], "xr")
        plt.plot(goal[0], goal[1], "xb")
        if ob is not None:
            plt.plot(ob[:, 0], ob[:, 1], "ok")
        plot_arrow(x[0], x[1], x[2])

        # # plot laser
        # T_robot = self.get_TF()
        # points = self.scan_to_points(self.last_obs[:self.pursuitor_model.laser_num], self.pursuitor_model.laser_min_angle,
        #                     self.pursuitor_model.laser_increment_angle, T_robot)
        # if len(points)>0:
        #     plt.plot(points[:,0], points[:,1], ".m")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
        # print(self.last_obs)
        return np.array([[[1, 1, 1]]
                         ], dtype=np.uint8)

    def scan_to_points(self, laser_ranges, min_angle, increment_angle, T_robot):
        laser_points = []
        for laser_index in range(len(laser_ranges)):
            laser_range = laser_ranges[laser_index]
            laser_angle = increment_angle * laser_index + min_angle
            # only plot reflected
            if laser_range < self.pursuitor_model.laser_max_range - 2:
                laser_points.append([laser_range * cos(laser_angle), laser_range * sin(laser_angle)])
        laser_points = np.array(laser_points)
        for i, laser_point in enumerate(laser_points):
            laser_point_tmp = np.matmul(T_robot, np.hstack((laser_point, 1)))
            laser_points[i] = laser_point_tmp[:2]
        return laser_points

    def _get_obs(self):
        """Returns the observation.
        """
        # laser_ranges = self.pursuitor_model.get_laser_scan(self.ob_list, self.ob_radius)
        percept_map = self.sample_map()
        percept_map = np.reshape(percept_map, (1, -1))[0].tolist()

        goal = self.get_goal()
        state = self.get_state()
        goal_angle = atan2(goal[1] - state[1], goal[0] - state[0]) - state[2]
        goal_angle = self.normlize_angle(goal_angle)
        goal_distance = np.linalg.norm(goal[:2] - state[:2])
        return np.array(state[3:5].tolist() + [goal_angle, goal_distance])

    def sample_map(self):
        """
        “exteroceptive information”, sample the percept region.
        :return: sampling points of percept region, with value representing occupancy (0 is non-free, 255 is free)
        """
        if self.ob_list is None:
            return np.ones([self.sample_map_width, self.sample_map_height])
        # represent obstacle in robot coordinate system
        T_robot = self.get_TF()
        ob_r_list = [np.matmul(T_robot.T, np.hstack((np.array(ob), 1)))[:2] for ob in self.ob_list]

        # filter obstacle out of percept region
        def filter_ob(obstacle_radius, ob_list, percept_region):
            percept_region_expanded = percept_region + np.array(
                [-obstacle_radius, obstacle_radius, -obstacle_radius, obstacle_radius])

            def is_in_percept_region(ob):
                check_ob_center_in = percept_region_expanded[0] <= ob[0] <= percept_region_expanded[1] and \
                                     percept_region_expanded[2] <= ob[1] <= percept_region_expanded[3]

                return check_ob_center_in

            filtered_ob_list = list(filter(is_in_percept_region, ob_list))
            return filtered_ob_list

        filtered_ob_list = filter_ob(self.ob_radius, ob_r_list, self.percept_region)
        if len(filtered_ob_list) is 0:
            return np.ones([self.sample_map_width, self.sample_map_height])
        sampled_map = np.zeros([self.sample_map_width, self.sample_map_height])
        for i in range(self.sample_map_width):
            for j in range(self.sample_map_height):
                sample_point = self.percept_sample_map[i][j]
                dis2ob = np.linalg.norm(np.array(filtered_ob_list) - sample_point, axis=1, keepdims=True)
                if np.any(dis2ob <= self.ob_radius):
                    sampled_map[i, j] = 0
                else:
                    sampled_map[i, j] = 1
        return sampled_map

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        u = self.get_state()[3:5]
        u[0] = self.max_v
        u[1] = (action[0] - (-1)) / 2 * (self.max_w+self.max_w) - self.max_w

        # use for differential guidence law
        self.last_pur_state = self.get_state()
        self.last_tar_state = self.get_goal()
        # u=[0,0]
        self.pursuitor_model.motion(u, self.dt)
        if self.evasion_mode:

            # if reach this point
            if np.linalg.norm(self.evasion_route[self.evasion_route_index, :2] - self.get_goal()[
                                                                                 :2]) <= self.robot_radius * 2:
                self.evasion_route_index = (self.evasion_route_index + 1) % len(self.evasion_route)
            target_u, _ = dwa_control(self.get_goal(), self.get_goal()[3:5], self.config_evasion,
                                      self.evasion_route[self.evasion_route_index, :2], self.ob_list)
            self.evader_model.motion((target_u), self.dt)
        else:
            self.evader_model.motion(self.target_u, self.dt)
        assert not self.robot_collision_with_obstacle(self.evader_model.state[:2], self.target_radius,
                                                      self.ob_list, self.ob_radius)
        x = self.get_state()
        self.traj = np.vstack((self.traj, x))  # store state history
        self.traj_pursuitor.append(self.get_state())
        self.pursuit_strategy.append(action)
        self.traj_evader.append(self.get_goal())

    def _compute_reward(self, observation):
        # reward = ((-observation[-1] + self.last_obs[-1])/(self.max_v*self.dt)) ** 3 * 10
        reward = -observation[-1] / 30.0 + -abs(observation[-2]) / 15.0

        if self.collision:
            reward = -150
        if self.catch:
            reward = 200
        return reward

    def _is_done(self, observations):
        self.catch = False
        self.collision = False
        if observations[-1] <= self.robot_radius + self.target_radius:
            self.catch = True
        #
        # laser_collision = np.array(observations[:self.pursuitor_model.laser_num]) <= self.robot_radius
        # if laser_collision.any():
        #     self.collision = True
        if self.robot_collision_with_obstacle(self.get_state()[:2], self.robot_radius, self.ob_list, self.ob_radius):
            self.collision = True
        return self.catch or self.collision

    def why_done(self):
        if self.catch:
            return 0
        elif self.collision:
            return 1
        elif self.record_count >= self.limit_step:
            return 2
        return False

    def robot_collision_with_obstacle(self, x, x_radius, ob_list, ob_radius):
        if ob_list is None:
            return False
        for ob in ob_list:
            if np.linalg.norm(x - ob) <= x_radius + ob_radius:
                return True
        return False

    def get_goal(self):
        return np.array(self.evader_model.state)

    def get_state(self):
        return np.array(self.pursuitor_model.state)

    def normlize_angle(self, angle):
        norm_angle = angle % 2 * math.pi
        if norm_angle > math.pi:
            norm_angle -= 2 * math.pi
        return norm_angle

    def get_TF(self):
        theta = self.get_state()[2]
        transition = self.get_state()[:2]
        T_robot = np.array([
            [cos(theta), -sin(theta), transition[0]],
            [sin(theta), cos(theta), transition[1]],
            [0, 0, 1]
        ])
        return T_robot

    def init_terrain(self, num):
        if num == 0:
            self.ob_list = np.array([[0, 2],
                                     [4.0, 2.0],
                                     [-2.0, 4.6],
                                     [5.0, 6.0],
                                     [7.0, 9.0],
                                     [12.0, 12.0],
                                     [4.0, 12.0]
                                     ])
        elif num == 1:
            self.ob_list = np.array([[0, 2],
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
        elif num == 2:
            self.ob_list = None

    def init_env_from_list(self, pursuit_init_num):
        if pursuit_init_num == 0:
            # pursuit env 0
            self.init_x = 2.1
            self.init_y = 7.5
            self.init_yaw = 0.0
        elif pursuit_init_num == 1:
            # pursuit env 1
            self.init_x = 8.0
            self.init_y = 6.0
            self.init_yaw = 0.0
        elif pursuit_init_num == 2:
            # pursuit env 2
            self.init_x = 12.0
            self.init_y = 8.0
            self.init_yaw = 0.0
        elif pursuit_init_num == 3:
            # pursuit env 3
            self.init_x = -3.0
            self.init_y = 0.0
            self.init_yaw = pi / 8.0

    def init_evader_circle(self, evader_init_num):
        if evader_init_num == 0:
            # evader env 1
            self.target_init_x = 0.0
            self.target_init_y = -10.0
            self.target_init_yaw = -pi / 2.0
            self.target_u = np.array([0.0, -pi / 5 * 2.0])
        elif evader_init_num == 1:
            # evader env 2
            self.target_init_x = 0.0
            self.target_init_y = 9.0
            self.target_init_yaw = pi - 0.3
            self.target_u = np.array([0.0, 0.66])
        elif evader_init_num == 2:
            # evader env 4
            self.target_init_x = 10.0
            self.target_init_y = 12.0
            self.target_init_yaw = -pi / 2.0
            self.target_u = np.array([0.0, -pi / 5])

    def init_evader_route(self, evader_init_num):
        if evader_init_num == 0:
            self.evasion_route = np.array([
                [4.0, 3.5],
                [6.0, 2.7],
                [7.55, 6.8],
                [9.57, 11.2],
                [9.3, 16.5],
                [5.44, 16.0],
                [5.44, 13.23],
                [5.44, 9.24],
                [0.0, 4.5],
                [-2.2, -1.1],
                [0.0, -5.0]
            ])
        elif evader_init_num == 1:
            self.evasion_route = np.array([
                [11.64, 8.78],
                [7.87, 10.94],
                [5.49, 12.91],
                [2.78, 13.79],
                [0.99, 10.71],
                [2.46, 3.27],
                [6.91, 2.53]
            ])
        elif evader_init_num == 2:
            self.evasion_route = np.array([
                [1.16, 8.15],
                [7.4, 5.63],
                [8.84, 0.78],
                [14.06, 11.86],
                [13.68, 17.4],
            ])
        self.evasion_route_index = 0
        self.target_init_x = self.evasion_route[0, 0]
        self.target_init_y = self.evasion_route[0, 1]
        self.target_init_yaw = math.atan2(self.evasion_route[1, 1] - self.evasion_route[0, 1],
                                          self.evasion_route[1, 0] - self.evasion_route[0, 0])

