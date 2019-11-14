import logging

import gym
from gym import spaces

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
from scipy.misc import imresize

logger = logging.getLogger(__name__)


class PursuitEnv2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.max_v = 2
        self.min_v = -0.3
        self.max_w = pi / 3.0 * 2
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
        self.ob_list = np.array([
            [0, 2],
            [4.0, 2.0],
            [5.0, 4.0],

            [5.0, 6.0],

            [7.0, 9.0],
            [12.0, 12.0]
        ])

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
        self.action_space = spaces.Discrete(self.action_num)

        # observations = image
        self.screen_height = 600 # pixel
        self.screen_width = 600 # pixel
        self.state_height = 150 # pixel
        self.state_width = 150 # pixel
        self.world_width = 20.0 # m

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.state_height, self.state_width, 3), dtype=np.uint8)
        # seed
        self.np_random = None
        self.seed()

        # time limit
        self.limit_step = 300
        self.count = 0

        self.action_plot = 0

        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.action_plot = action
        self._set_action(action)

        observation = self._get_obs()
        done = self._is_done(observation)
        if self.count >= self.limit_step:
            done = True
        reward = self._compute_reward(observation)
        self.last_obs = observation
        self.count += 1
        return observation, reward, done, {}

    def reset(self):
        self.init_env_from_list(random.randint(0, 3), random.randint(1, 2))
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
        return obs

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        # render image

        scale = self.screen_width / self.world_width
        origin = [self.screen_width / 10, self.screen_height / 10]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            # plot robot with orientation
            pursuit_radius = int(self.robot_radius * scale)
            pursuit = rendering.make_polygon(
                [(-pursuit_radius, pursuit_radius), (0, pursuit_radius), (pursuit_radius, 0),
                 (0, -pursuit_radius), (-pursuit_radius, -pursuit_radius)], filled=True)
            pursuit.set_color(.8, .6, .4)
            self.pursuit_trans = rendering.Transform()
            pursuit.add_attr(self.pursuit_trans)
            self.viewer.add_geom(pursuit)

            # plot target with orientation
            evader_radius = int(self.robot_radius * scale)
            evader = rendering.make_polygon([(-evader_radius, evader_radius), (0, evader_radius), (evader_radius, 0),
                                             (0, -evader_radius), (-evader_radius, -evader_radius)])
            evader.set_color(.5, .5, .8)
            self.evader_trans = rendering.Transform()
            evader.add_attr(self.evader_trans)
            self.viewer.add_geom(evader)

            # plot obstacles
            for ob in self.ob_list:
                ob_radius = self.ob_radius * scale
                obstacle = rendering.make_circle(ob_radius)
                obstacle.set_color(0, 0, 0)
                obstacle.add_attr(
                    rendering.Transform(translation=(ob[0] * scale + origin[0], ob[1] * scale + origin[1])))
                self.viewer.add_geom(obstacle)

        self.pursuit_trans.set_translation(self.get_state()[0] * scale + origin[0],
                                           self.get_state()[1] * scale + origin[1])
        self.pursuit_trans.set_rotation(self.get_state()[2])

        self.evader_trans.set_translation(self.get_goal()[0] * scale + origin[0],
                                          self.get_goal()[1] * scale + origin[1])
        self.evader_trans.set_rotation(self.get_goal()[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


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

        arr = self.render(mode='rgb_array')
        arr = imresize(arr, (self.state_height, self.state_width, 3))
        return arr


    def sample_map(self):
        """
        “exteroceptive information”, sample the percept region.
        :return: sampling points of percept region, with value representing occupancy (0 is non-free, 255 is free)
        """
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

        sampled_map = np.zeros([self.sample_map_width, self.sample_map_height])
        for i in range(self.sample_map_width):
            for j in range(self.sample_map_height):
                sample_point = self.percept_sample_map[i][j]
                dis2ob = np.linalg.norm(np.array(filtered_ob_list) - sample_point, axis=1, keepdims=True)
                if np.any(dis2ob <= self.ob_radius):
                    sampled_map[i, j] = 0
                else:
                    sampled_map[i, j] = 255
        return sampled_map

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        u = [0, 0]

        if action == 0:
            u, ltraj = dwa_control(self.get_state(), np.array(self.pursuitor_model.state[3:5]), self.config_dwa,
                                   self.get_goal()[:2], self.ob_list)

        elif action == 1:
            # u = apf_control(self.confg_apf, self.pursuitor_model.state, self.evader_model.state, self.ob_list)
            #
            # # get angle velocity from angle input
            # u[1] = self.pursuitor_model.rot_to_angle(u[1])
            w = pn_control(self.confg_pn, self.get_state(), self.get_goal(), self.last_pur_state, self.last_tar_state)
            u = [self.max_v, w]

        self.last_pur_state = self.get_state()
        self.last_tar_state = self.get_goal()
        self.pursuitor_model.motion(u, self.dt)

        self.evader_model.motion(self.target_u, self.dt)
        assert not self.robot_collision_with_obstacle(self.evader_model.state[:2], self.target_radius,
                                                      self.ob_list, self.ob_radius)
        x = self.get_state()
        self.traj = np.vstack((self.traj, x))  # store state history

    def _compute_reward(self, observation):
        # reward = ((-observation[-1] + self.last_obs[-1])/(self.max_v*self.dt)) ** 3 * 10
        reward = -observation[-1] / 10.0

        if self.collision:
            reward = -150
        if self.catch:
            reward = 200
        return reward

    def _is_done(self, observations):
        self.catch = False
        self.collision = False

        if np.linalg.norm(self.get_goal()[:2]-self.get_state()[:2]) <= self.robot_radius + self.target_radius:
            self.catch = True
        #
        # laser_collision = np.array(observations[:self.pursuitor_model.laser_num]) <= self.robot_radius
        # if laser_collision.any():
        #     self.collision = True
        if self.robot_collision_with_obstacle(self.get_state()[:2], self.robot_radius, self.ob_list, self.ob_radius):
            self.collision = True
        return self.catch or self.collision

    def robot_collision_with_obstacle(self, x, x_radius, ob_list, ob_radius):
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

    def init_env_from_list(self, pursuit_init_num, evader_init_num):
        if pursuit_init_num == 0:
            # pursuit env 0
            self.init_x = 0.0
            self.init_y = 6.0
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
            self.init_x = 0.0
            self.init_y = 0.0
            self.init_yaw = pi / 8.0

        # if evader_init_num==0:
        #     # evader env 0
        #     self.target_init_x = 2.0
        #     self.target_init_y = 11.0
        #     self.target_init_yaw = -pi/2.0
        #     self.target_u = np.array([1.8, 0.0])
        if evader_init_num == 1:
            # evader env 1
            self.target_init_x = 0.0
            self.target_init_y = 10.0
            self.target_init_yaw = -pi / 2.0
            self.target_u = np.array([1.8, -pi / 5 * 2.0])
        elif evader_init_num == 2:
            # evader env 2
            self.target_init_x = 0.0
            self.target_init_y = 9.0
            self.target_init_yaw = pi - 0.3
            self.target_u = np.array([1.8, 0.66])
            # elif evader_init_num==3:
            #     # evader env 3
            #     self.target_init_x = 10.0
            #     self.target_init_y = 15.0
            #     self.target_init_yaw = -pi/2.0
            #     self.target_u = np.array([1.8, 0.0])


