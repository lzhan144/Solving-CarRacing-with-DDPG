import torch
import gym
import numpy as np
from TD3_image import TD3
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
import utils
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
    log_interval = 1           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 0.1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 1000         # max num of episodes
    max_timesteps = 2000        # max timesteps in one episode
    save_every = 100            # model saving interal
    img_stack = 4               # number of image stacks together
    action_repeat = 8           # repeat action in N frames
    max_size = 1e6
    
    
    """ beta Prioritized Experience Replay"""
    beta_start = 0.4
    beta_frames = 25000
    
    if not os.path.exists('./TD3tested'):
        os.mkdir('./TD3tested')
    directory = "./TD3tested/{}".format(env_name) # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    render = True
    save_gif = True
    ###################################
    
    env = Env(env_name, random_seed, img_stack, action_repeat)
    action_dim = env.action_space.shape[0]
    
    policy = TD3(action_dim, img_stack)
    replay_buffer = NaivePrioritizedBuffer(int(max_size))
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open("log.txt","w+")
    
    
    ## for plot
    Reward = []
    total_timesteps = 0
    episode_timesteps = 0
    
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        episode_timesteps = 0
        for t in range(max_timesteps):
            # select action and add exploration noise:
#             print("state: " + str(state))
            action = policy.select_action(state)          
            action = action + np.random.normal(0, exploration_noise, size = action_dim)
            action = action.clip(env.action_space.low, env.action_space.high)
#             print("action clipped: " + str(action))
            
            # take action in env:
            next_state, reward, done, _ = env.step(action)
#             print("state: " +str(next_state))
            replay_buffer.add(state, next_state, action, reward, float(done))
            state = next_state
            
            avg_reward += reward
            ep_reward += reward
            total_timesteps += 1
            episode_timesteps += 1
                    
            # if episode is done then update policy:
            if done or t==(max_timesteps-1):
                beta = min(1.0, beta_start + total_timesteps * (1.0 - beta_start) / beta_frames)
                policy.train(replay_buffer, episode_timesteps, beta)     
                break
        
        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        
        Reward.append(ep_reward)
        
        ep_reward = 0
        
        # if avg reward > 300 then save and stop traning:
        if avg_reward >= 900:
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

        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0
            
    plt.plot(range(len(Reward)),np.array(Reward), 'b')
    plt.savefig('./TD3tested/episode reward.png')  

#         plt.plot(range(len(policy.actor_loss)), policy.actor_loss)
    plt.plot(range(len(policy.actor_loss)),np.array(policy.actor_loss),'b')
    plt.savefig('./TD3tested/actor loss.png')

    plt.plot(range(len(policy.critic_loss1)),np.array(policy.critic_loss1),'b')
    plt.savefig('./TD3tested/critic loss1.png')

    plt.plot(range(len(policy.critic_loss2)),np.array(policy.critic_loss2),'b')
    plt.savefig('./TD3tested/critic loss2.png')
    
    
### main function 
train('CarRacing-v0')