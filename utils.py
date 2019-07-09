import numpy as np
import visdom
import torch
import torch.nn as nn

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
	def __init__(self, max_size=1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, state, new_state, action, reward, done_bool):
		data = (state, new_state, action, reward, done_bool)
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind: 
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class NaivePrioritizedBuffer(object):
	def __init__(self, capacity, prob_alpha=0.6):
		self.prob_alpha = prob_alpha
		self.capacity = capacity
		self.buffer = []
		self.pos = 0
		self.priorities = np.zeros((capacity,), dtype=np.float32)

	def add(self, state, next_state, action, reward, done):
		state = state.numpy()
		next_state = next_state.numpy()
		assert state.ndim == next_state.ndim
		state = np.expand_dims(state, 0)
		next_state = np.expand_dims(next_state, 0)

		max_prio = self.priorities.max() if self.buffer else 1.0

		if len(self.buffer) < self.capacity:
			self.buffer.append((state, next_state, action, reward, done))
		else:
			self.buffer[self.pos] = (state, next_state, action, reward, done)

		self.priorities[self.pos] = max_prio
		self.pos = (self.pos + 1) % self.capacity

	def sample(self, batch_size, beta=0.4):
		if len(self.buffer) == self.capacity:
			prios = self.priorities
		else:
			prios = self.priorities[:self.pos]

		probs = prios ** self.prob_alpha
		probs /= probs.sum()

		indices = np.random.choice(len(self.buffer), batch_size, p=probs)
		samples = [self.buffer[idx] for idx in indices]

		total = len(self.buffer)
		weights = (total * probs[indices]) ** (-beta)
		weights /= weights.max()
		weights = np.array(weights, dtype=np.float32)

		batch = list(zip(*samples))
		states = np.concatenate(batch[0])
		actions = batch[2]
		rewards = batch[3]
		next_states = np.concatenate(batch[1])
		dones = batch[4]

		return np.array(states), np.array(next_states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1), indices, weights

	def update_priorities(self, batch_indices, batch_priorities):
		for idx, prio in list(zip(batch_indices, batch_priorities)):
			self.priorities[idx] = prio

	def __len__(self):
		return len(self.buffer)


class DrawLine():
	def __init__(self, env, title, xlabel=None, ylabel=None):
		self.vis = visdom.Visdom()
		self.update_flag = False
		self.env = env
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.title = title

	def __call__(
			self,
			xdata,
			ydata,
    ):
		if not self.update_flag:
			self.win = self.vis.line(
				X=np.array([xdata]),
				Y=np.array([ydata]),
				opts=dict(
					xlabel=self.xlabel,
					ylabel=self.ylabel,
					title=self.title,
				),
				env=self.env,
			)
			self.update_flag = True
		else:
			self.vis.line(
				X=np.array([xdata]),
				Y=np.array([ydata]),
				win=self.win,
				env=self.env,
				update='append',
			)