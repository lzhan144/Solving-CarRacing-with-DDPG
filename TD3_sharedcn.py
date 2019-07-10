import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
import math
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

feat_size = 2
latent_dim = 256

''' Utilities '''

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Net(nn.Module):
    """
    Actor-Critic network for TD3
    """
    def __init__(self, action_dim, img_stack):
        super(Net, self).__init__()
        self.encoder = torch.nn.ModuleList([  ## input size:[96, 96]
            torch.nn.Conv2d(img_stack, 16, 5, 4, padding=2),  ## output size: [16, 24, 24]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 4, padding=2),  ## output size: [32, 6, 6]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 3, 3]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 2, padding=2),  ## output size: [128, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 1, 1]
            Flatten(),
        ])


            # torch.nn.Conv2d(img_stack, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(16),
            # torch.nn.Conv2d(16, 32, 5, 2, padding=2),  ## output size: [32, 24, 24]
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(32),
            # torch.nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 12, 12]
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(64),
            # torch.nn.Conv2d(64, 128, 5, 2, padding=2),  ## output size: [128, 6, 6]
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 3, 3]
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(256),
            # torch.nn.Conv2d(256, 512, 5, 2, padding=2),  ## output size: [512, 1, 1]
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(512),


        self.actor = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim, 30),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(30),
        ])

        self.action_out_1 = torch.nn.Linear(30, int(action_dim / 3))
        self.action_out_2 = torch.nn.Linear(30, int(action_dim / 3) * 2)


        self.critic_1 = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + action_dim, 30),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(30),
            torch.nn.Linear(30, 1),
        ])

        self.critic_2 = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + action_dim, 30),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(30),
            torch.nn.Linear(30, 1),
        ])


    def forward(self, x, u):

        for layer in self.encoder:
            x = layer(x)
            # print("x size: " + str(x.size()))

        ## actor branch
        action = x
        for layer in self.actor:
            action = layer(action)
        action_out1 = torch.tanh(self.action_out_1(action))
        action_out2 = torch.sigmoid(self.action_out_2(action))

        action_out = torch.cat((action_out1, action_out2), 1)

        ## critic branch
        x = torch.cat([x,u],1)
        value_1 = x
        for layer in self.critic_1:
            value_1 = layer(value_1)
        value_2 = x
        for layer in self.critic_2:
            value_2 = layer(value_2)

        return action_out, value_1, value_2

    def Q1(self, x, u):
        for layer in self.encoder:
            x = layer(x)
        x = torch.cat([x, u], 1)
        value_1 = x
        for layer in self.critic_1:
            value_1 = layer(value_1)

        return value_1


class TD3(object):

    def __init__(self, action_dim, img_stack):

        self.action_dim = action_dim
        self.net = Net(action_dim, img_stack).to(device)
        self.net_target = Net(action_dim, img_stack).to(device)
        self.net_target.load_state_dict(self.net.state_dict())
        self.net_optimizer = torch.optim.Adam(self.net.parameters())


        # self.max_action = max_action

    def select_action(self, state):
        state = state.float().to(device)
        fake_action = torch.FloatTensor(np.zeros((1,self.action_dim))).to(device)  ## hold the position
        action,_,_ = self.net(state, fake_action)
        return action.cpu().data.numpy().flatten()


    def train(self, replay_buffer, iterations, beta_PER, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        for it in range(iterations):
            # Sample replay buffer
            x, y, u, r, d, indices, w = replay_buffer.sample(batch_size, beta=beta_PER)
            state = torch.FloatTensor(x).squeeze(1).to(device)
            #             print('state size: ' +str(state.size()))
            u = u.reshape((batch_size, self.action_dim))
            action = torch.FloatTensor(u).to(device)
            #             print('action size: ' +str(action.size()))
            next_state = torch.FloatTensor(y).squeeze(1).to(device)
            #             print('next state size: ' +str(next_state.size()))
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)
            w = w.reshape((batch_size, -1))
            weights = torch.FloatTensor(w).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action,_,_ = self.net_target(next_state, action)  ## input action is not used
            next_action = (next_action + noise).clamp(-1, 1)
            #print(next_action)

            # Compute the target Q value
            _,target_Q1, target_Q2 = self.net_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            _,current_Q1, current_Q2 = self.net(state, action)

            # Compute critic loss
            critic_loss = weights * ((current_Q1 - target_Q).pow(2) + (current_Q2 - target_Q).pow(2))
            prios = critic_loss + 1e-5
            critic_loss = critic_loss.mean()

            # Optimize the critic
            self.net_optimizer.zero_grad()
            critic_loss.backward()
            replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
            self.net_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss

                action_Q1,_,_ = self.net(state, torch.FloatTensor(np.zeros((batch_size,self.action_dim))).to(device))
                actor_loss = - self.net.Q1(state, action_Q1).mean()

                # Optimize the actor
                self.net_optimizer.zero_grad()
                actor_loss.backward()
                self.net_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.net.parameters(), self.net_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, directory, name):
        torch.save(self.net.state_dict(), '%s/%s_net.pth' % (directory, name))
        torch.save(self.net_target.state_dict(), '%s/%s_net_target.pth' % (directory, name))


    def load(self, directory, name):
        self.net.load_state_dict(
            torch.load('%s/%s_net.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.net_target.load_state_dict(
            torch.load('%s/%s_net_target.pth' % (directory, name), map_location=lambda storage, loc: storage))


