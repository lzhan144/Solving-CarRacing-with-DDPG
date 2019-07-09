import torch
import gym
import numpy as np
from TD3 import TD3
from utils import NaivePrioritizedBuffer
import os
import roboschool, gym
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import Box2D

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import DrawLine
import math

from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, env_name, random_seed, img_stack, action_repeat):
        self.env = gym.make(env_name)
        self.env.seed(random_seed)
        self.action_space = self.env.action_space
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = img_stack
        self.action_repeat = action_repeat

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        #         print(img_rgb)
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [np.expand_dims(img_gray, axis=0)] * self.img_stack  # four frames for decision
        return torch.FloatTensor(self.stack).permute(1, 0, 2, 3)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(np.expand_dims(img_gray, axis=0))
        assert len(self.stack) == self.img_stack
        return torch.FloatTensor(self.stack).permute(1, 0, 2, 3), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


def train(env):
    ######### Hyperparameters #########
    env_name = env
    log_interval = 10  # print avg reward after interval
    random_seed = 0
    gamma = 0.99  # discount for future rewards
    batch_size = 100  # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 0.1
    polyak = 0.995  # target policy update parameter (1-tau)
    policy_noise = 0.2  # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2  # delayed policy updates parameter
    max_episodes = int(1e6)  # max num of episodes
    max_timesteps = 2000  # max timesteps in one episode
    save_every = 100  # model saving interal
    img_stack = 4  # number of image stacks together
    action_repeat = 8  # repeat action in N frames
    max_size = 1e6
    vis = True

    """ beta Prioritized Experience Replay"""
    beta_start = 0.4
    beta_frames = 25000

    # if not os.path.exists('./TD3tested'):
    #     os.mkdir('./TD3tested')
    directory = "./{}".format(env_name)  # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)

    ###################################

    env = Env(env_name, random_seed, img_stack, action_repeat)
    action_dim = env.action_space.shape[0]
    # if vis:
    #     draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")

    policy = TD3(action_dim, img_stack)
    replay_buffer = NaivePrioritizedBuffer(int(max_size))

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)

    # logging variables:

    log_f = open("log.txt", "w+")
    ## for plot
    Reward = []
    total_timesteps = 0
    episode_timesteps = 0
    running_score = 0

    # training procedure:
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_timesteps = 0
        score = 0

        for t in range(max_timesteps):
            # select action and add exploration noise:
            #             print("state: " + str(state))
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=action_dim)
            action = action.clip(env.action_space.low, env.action_space.high)
            #             print("action clipped: " + str(action))

            # take action in env:
            next_state, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            #             print("state: " +str(next_state))
            env.render()
            replay_buffer.add(state, next_state, action, reward, float(done))
            state = next_state

            score += reward
            total_timesteps += 1
            episode_timesteps += 1

            # if episode is done then update policy:
            if done or t == (max_timesteps - 1):
                beta = min(1.0, beta_start + total_timesteps * (1.0 - beta_start) / beta_frames)
                policy.train(replay_buffer, episode_timesteps, beta)
                break

        running_score = running_score * 0.99 + score * 0.01



        if episode % log_interval == 0:
            # if vis:
            #     draw_reward(xdata = episode, ydata = running_score)
            log_f.write('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(episode, score, running_score))
            log_f.flush()
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(episode, score, running_score))


        # if avg reward > 300 then save and stop traning:
        if running_score >= 900:
            #         if episode % save_every == 0:
            print("########## Model received ###########")
            name = filename
            policy.save(directory, name)
            log_f.close()
            break

        if episode % 100 == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)
            policy.save(directory, filename)


### main function
train('CarRacing-v0')