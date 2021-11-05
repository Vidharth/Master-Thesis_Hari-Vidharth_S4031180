__author__ = "Hari Vidharth"
__date__ = "July 2020"


import gc
import time

import gym
import wandb
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend

from oracle.oracle import demonstrations


tf.config.set_visible_devices([], 'GPU')


class CriticNetwork(Model):
	def __init__(self):
		super(CriticNetwork, self).__init__()

		w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

		self.fc1 = Dense(256, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.fc2 = Dense(256, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.fc3 = Dense(256, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.q = Dense(1, activation="linear", kernel_initializer=w_init, kernel_regularizer="l2")

	def call(self, state, action):
		x = self.fc1(tf.concat([state, action], axis=1))
		x = self.fc2(x)
		x = self.fc3(x)
		q = self.q(x)

		return q


class ActorNetwork(Model):
	def __init__(self, action_shape):
		super(ActorNetwork, self).__init__()

		w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

		self.fc1 = Dense(256, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.fc2 = Dense(256, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.fc3 = Dense(256, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.a = Dense(action_shape, activation='tanh', kernel_initializer=w_init, kernel_regularizer="l2")

	def call(self, state):
		x = self.fc1(state)
		x = self.fc2(x)
		x = self.fc3(x)
		a = self.a(x)

		return a


class Agent():
	def __init__(self, env):
		self.actor = ActorNetwork(env.action_space.shape[0])
		self.critic_1 = CriticNetwork()
		self.critic_2 = CriticNetwork()
		self.target_actor = ActorNetwork(env.action_space.shape[0])
		self.target_critic_1 = CriticNetwork()
		self.target_critic_2 = CriticNetwork()

	def test_choose_action(self, state):
		state = tf.convert_to_tensor([observation], dtype=tf.float32)
		action = self.actor(state)[0]
		action = tf.clip_by_value(action, env.action_space.low[0], env.action_space.high[0])

		return action

	def load_networks(self):
		print(" ")
		print('... loading networks ...')
		print(" ")

		self.actor.load_weights("./checkpoints/Actor").expect_partial()
		self.critic_1.load_weights("./checkpoints/Critic1").expect_partial()
		self.critic_2.load_weights("./checkpoints/Critic2").expect_partial()
		self.target_actor.load_weights("./checkpoints/TargetActor").expect_partial()
		self.target_critic_1.load_weights("./checkpoints/TargetCritic1").expect_partial()
		self.target_critic_2.load_weights("./checkpoints/TargetCritic2").expect_partial()


if __name__ == "__main__":

	env = gym.make("FetchPush-v1", reward_type="dense")

	state = env.reset()
	observation = state["observation"]
	desired_goal = state["desired_goal"]
	action, o_mean, o_std, g_mean, g_std = demonstrations(env, observation, desired_goal)
	env.close()

	agent = Agent(env)
	agent.load_networks()

	accuracy = []

	for test in range(10):
		state = env.reset()
		done = False

		while not done:
			env.render()
			o_clip = np.clip(state["observation"], -200, 200)
			g_clip = np.clip(state["desired_goal"], -200, 200)
			o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
			g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
			observation = np.concatenate([o_norm, g_norm])

			action = agent.test_choose_action(observation)

			next_state, extrinsic_reward, done, info = env.step(action)

			state = next_state

		if info["is_success"] == 1.0:
			accuracy.append(1.0)

	# print("ACCURACY: ", accuracy.count(1.0))
