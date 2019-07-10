import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

feat_size = 1
latent_dim = 512

''' Utilities '''


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Actor(nn.Module):
    def __init__(self, action_dim, img_stack):
        super(Actor, self).__init__()

        self.encoder = torch.nn.ModuleList([  ## input size:[96, 96]
            torch.nn.Conv2d(img_stack, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 2, padding=2),  ## output size: [32, 24, 24]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 12, 12]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),  ## output size: [128, 3, 3]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, 5, 2, padding=2),  ## output size: [512, 1, 1]
            Flatten(),  ## output: 512
        ])

        self.linear = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, action_dim),
            torch.nn.Tanh(),
        ])

    def forward(self, x):

        for layer in self.encoder:
            x = layer(x)
        # print(x.size())
        for layer in self.linear:
            x = layer(x)
            # print(x.size())

        return x

class Critic(nn.Module):
    def __init__(self, action_dim, img_stack):
        super(Critic, self).__init__()

        self.encoder = torch.nn.ModuleList([  ## input size:[96, 96]
            torch.nn.Conv2d(img_stack, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 2, padding=2),  ## output size: [32, 24, 24]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 12, 12]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),  ## output size: [128, 3, 3]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, 5, 2, padding=2),  ## output size: [512, 1, 1]
            Flatten(),  ## output: 512
        ])

        self.linear = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + action_dim, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),
        ])

    def forward(self, x, u):

        for layer in self.encoder:
            x = layer(x)
        counter = 0
        for layer in self.linear:
            counter += 1
            if counter == 1:
                x = torch.cat([x, u], 1)
                x = layer(x)
            else:
                x = layer(x)

        return x



class DDPG(object):
	def __init__(self, action_dim, img_stack):
		self.action_dim = action_dim
		self.actor = Actor(action_dim, img_stack).to(device)
		self.actor_target = Actor(action_dim, img_stack).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
		self.actor_loss = []

		self.critic = Critic(action_dim, img_stack).to(device)
		self.critic_target = Critic(action_dim, img_stack).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
		self.critic_loss = []


	def select_action(self, state):
		state = state.float().to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, beta_PER, batch_size=100, discount=0.99, tau=0.005):

		for it in range(iterations):

			# print("training")

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

			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action)

			# Compute critic loss
			critic_loss = weights * ((current_Q - target_Q).pow(2))
			prios = critic_loss + 1e-5
			critic_loss = critic_loss.mean()
			self.critic_loss.append(critic_loss)
			# print("critic_loss"+str(critic_loss))

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()
			self.actor_loss.append(actor_loss)
			# print("actor_loss"+ str(actor_loss))

			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

	def save(self, directory, name):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
		torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))

		torch.save(self.critic.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
		torch.save(self.critic_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))

	def load(self, directory, name):
		self.actor.load_state_dict(
			torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
		self.actor_target.load_state_dict(
			torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

		self.critic.load_state_dict(
			torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
		self.critic_target.load_state_dict(
			torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

	def load_actor(self, directory, name):
		self.actor.load_state_dict(
			torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
		self.actor_target.load_state_dict(
			torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))


