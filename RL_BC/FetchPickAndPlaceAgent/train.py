__author__ = "Hari Vidharth"
__date__ = "July 2020"


import gc
import time

import gym
import wandb
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend

from oracle.oracle import demonstrations


gc.collect()
backend.clear_session()


class Memory():
	def __init__(self, size, state_shape, action_shape):
		self.size = size
		self.memory_counter = 0

		self.state_memory = np.zeros((self.size, *state_shape))
		self.action_memory = np.zeros((self.size, action_shape))
		self.reward_memory = np.zeros(self.size)
		self.next_state_memory = np.zeros((self.size, *state_shape))
		self.terminal_memory = np.zeros(self.size, dtype=bool)

	def store_transition(self, state, action, reward, next_state, done):
		index = self.memory_counter % self.size

		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.next_state_memory[index] = next_state
		self.terminal_memory[index] = done

		self.memory_counter += 1

	def sample_buffer(self, batch_size):
		sample_size = min(self.memory_counter, self.size)
		batch = np.random.choice(sample_size, batch_size)

		states = self.state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		next_states = self.next_state_memory[batch]
		dones = self.terminal_memory[batch]

		return states, actions, rewards, next_states, dones


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
		self.learn_step_counter = 0

		self.agent_memory = Memory(1000000, (28, ), env.action_space.shape[0])
		self.demo_memory = Memory(2500, (28, ), env.action_space.shape[0])

		self.actor = ActorNetwork(env.action_space.shape[0])
		self.critic_1 = CriticNetwork()
		self.critic_2 = CriticNetwork()
		self.target_actor = ActorNetwork(env.action_space.shape[0])
		self.target_critic_1 = CriticNetwork()
		self.target_critic_2 = CriticNetwork()

		self.actor.compile(optimizer="adam")
		self.critic_1.compile(optimizer="adam")
		self.critic_2.compile(optimizer="adam")
		self.target_actor.compile(optimizer="adam")
		self.target_critic_1.compile(optimizer="adam")
		self.target_critic_2.compile(optimizer="adam")

		self.update_network_parameters(tau=1)

	def train_choose_action(self, state):
		state = tf.convert_to_tensor([observation], dtype=tf.float32)
		action = self.actor(state)[0]
		action = action + np.random.normal(scale=0.1)
		action = tf.clip_by_value(action, env.action_space.low[0], env.action_space.high[0])

		return action

	def test_choose_action(self, state):
		state = tf.convert_to_tensor([observation], dtype=tf.float32)
		action = self.actor(state)[0]
		action = tf.clip_by_value(action, env.action_space.low[0], env.action_space.high[0])

		return action

	def agent_remember(self, state, action, reward, next_state, done):
		self.agent_memory.store_transition(state, action, reward, next_state, done)

	def demo_remember(self, state, action, reward, next_state, done):
		self.demo_memory.store_transition(state, action, reward, next_state, done)

	def update_network_parameters(self, tau=None):
		if tau is None:
			tau = 0.005

		weights = []
		targets = self.target_actor.weights
		for i, weight in enumerate(self.actor.weights):
			weights.append(weight * tau + targets[i]*(1-tau))
		self.target_actor.set_weights(weights)

		weights = []
		targets = self.target_critic_1.weights
		for i, weight in enumerate(self.critic_1.weights):
			weights.append(weight * tau + targets[i]*(1-tau))
		self.target_critic_1.set_weights(weights)

		weights = []
		targets = self.target_critic_2.weights
		for i, weight in enumerate(self.critic_2.weights):
			weights.append(weight * tau + targets[i]*(1-tau))
		self.target_critic_2.set_weights(weights)

	def save_networks(self):
		print(" ")
		print('... saving networks ...')
		print(" ")

		self.actor.save_weights("./checkpoints/Actor")
		self.critic_1.save_weights("./checkpoints/Critic1")
		self.critic_2.save_weights("./checkpoints/Critic2")
		self.target_actor.save_weights("./checkpoints/TargetActor")
		self.target_critic_1.save_weights("./checkpoints/TargetCritic1")
		self.target_critic_2.save_weights("./checkpoints/TargetCritic2")

	def demo_learn(self):
		states, actions, rewards, next_states, dones = self.demo_memory.sample_buffer(128)

		states = tf.convert_to_tensor(states, dtype=tf.float32)
		actions = tf.convert_to_tensor(actions, dtype=tf.float32)
		rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
		next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

		target_actions = self.target_actor(next_states)
		target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -50, 0.5)
		target_actions = tf.clip_by_value(target_actions, env.action_space.low[0], env.action_space.high[0])
		q1_ = tf.squeeze(self.target_critic_1(next_states, target_actions), 1)
		q2_ = tf.squeeze(self.target_critic_2(next_states, target_actions), 1)
		critic_value_ = tf.minimum(q1_, q2_)
		target = rewards + 0.98*critic_value_*(1-dones)

		with tf.GradientTape() as critic_1_tape:
			critic_1_tape.watch(self.critic_1.trainable_variables)
			q1 = tf.squeeze(self.critic_1(states, actions), 1)
			critic_1_loss = tf.reduce_mean(tf.square(tf.stop_gradient(target) - q1))
			critic_1_loss += tf.reduce_sum(self.critic_1.losses)
		critic_1_gradient = critic_1_tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
		self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
		del critic_1_tape

		with tf.GradientTape() as critic_2_tape:
			critic_2_tape.watch(self.critic_2.trainable_variables)
			q1 = tf.squeeze(self.critic_2(states, actions), 1)
			critic_2_loss = tf.reduce_mean(tf.square(tf.stop_gradient(target) - q1))
			critic_2_loss += tf.reduce_sum(self.critic_2.losses)
		critic_2_gradient = critic_2_tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
		self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))
		del critic_2_tape

		self.learn_step_counter += 1

		if self.learn_step_counter % 2 == 0:
			new_actions = self.actor(states)
			q1 = tf.squeeze(self.critic_1(states, new_actions), 1)
			q2 = tf.squeeze(self.critic_2(states, new_actions), 1)
			critic_p_value = tf.minimum(q1, q2)
			q1 = tf.squeeze(self.critic_1(states, actions), 1)
			q2 = tf.squeeze(self.critic_2(states, actions), 1)
			critic_value = tf.minimum(q1, q2)
			mask = critic_p_value > critic_value
			filtered_demo_actions = tf.boolean_mask(actions, mask)
			filtered_actor_actions = tf.boolean_mask(new_actions, mask)

			with tf.GradientTape() as actor_tape:
				actor_tape.watch(self.actor.trainable_variables)
				new_actions = self.actor(states)
				q1 = tf.squeeze(self.critic_1(states, new_actions), 1)
				q2 = tf.squeeze(self.critic_2(states, new_actions), 1)
				critic_value = tf.minimum(q1, q2)
				actor_loss = -tf.reduce_mean(critic_value)
				actor_loss += tf.reduce_sum(self.actor.losses)
				actor_loss += 2.0 * tf.reduce_sum(tf.square(filtered_demo_actions - filtered_actor_actions))
			actor_gradient = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
			self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
			del actor_tape

			self.update_network_parameters()

			wandb.log({"Critic 1 Loss": critic_1_loss.numpy()})
			wandb.log({"Critic 2 Loss": critic_2_loss.numpy()})
			wandb.log({"Actor Loss": actor_loss.numpy()})

	def learn(self):
		demo_states, demo_actions, demo_rewards, demo_next_states, demo_dones = self.demo_memory.sample_buffer(128)
		states, actions, rewards, next_states, dones = self.agent_memory.sample_buffer(1024)

		demo_states = tf.convert_to_tensor(demo_states, dtype=tf.float32)
		demo_actions = tf.convert_to_tensor(demo_actions, dtype=tf.float32)
		demo_rewards = tf.convert_to_tensor(demo_rewards, dtype=tf.float32)
		demo_next_states = tf.convert_to_tensor(demo_next_states, dtype=tf.float32)

		states = tf.convert_to_tensor(states, dtype=tf.float32)
		actions = tf.convert_to_tensor(actions, dtype=tf.float32)
		rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
		next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

		target_actions = self.target_actor(next_states)
		target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -50, 0.5)
		target_actions = tf.clip_by_value(target_actions, env.action_space.low[0], env.action_space.high[0])
		q1_ = tf.squeeze(self.target_critic_1(next_states, target_actions), 1)
		q2_ = tf.squeeze(self.target_critic_2(next_states, target_actions), 1)
		critic_value_ = tf.minimum(q1_, q2_)
		target = rewards + 0.98*critic_value_*(1-dones)

		with tf.GradientTape() as critic_1_tape:
			critic_1_tape.watch(self.critic_1.trainable_variables)
			q1 = tf.squeeze(self.critic_1(states, actions), 1)
			critic_1_loss = tf.reduce_mean(tf.square(tf.stop_gradient(target) - q1))
			critic_1_loss += tf.reduce_sum(self.critic_1.losses)
		critic_1_gradient = critic_1_tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
		self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
		del critic_1_tape

		with tf.GradientTape() as critic_2_tape:
			critic_2_tape.watch(self.critic_2.trainable_variables)
			q1 = tf.squeeze(self.critic_2(states, actions), 1)
			critic_2_loss = tf.reduce_mean(tf.square(tf.stop_gradient(target) - q1))
			critic_2_loss += tf.reduce_sum(self.critic_2.losses)
		critic_2_gradient = critic_2_tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
		self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))
		del critic_2_tape

		self.learn_step_counter += 1

		if self.learn_step_counter % 2 == 0:
			new_actions = self.actor(demo_states)
			q1 = tf.squeeze(self.critic_1(demo_states, new_actions), 1)
			q2 = tf.squeeze(self.critic_2(demo_states, new_actions), 1)
			critic_p_value = tf.minimum(q1, q2)
			q1 = tf.squeeze(self.critic_1(demo_states, demo_actions), 1)
			q2 = tf.squeeze(self.critic_2(demo_states, demo_actions), 1)
			critic_value = tf.minimum(q1, q2)
			mask = critic_p_value > critic_value
			filtered_demo_actions = tf.boolean_mask(demo_actions, mask)
			filtered_actor_actions = tf.boolean_mask(new_actions, mask)

			with tf.GradientTape() as actor_tape:
				actor_tape.watch(self.actor.trainable_variables)
				new_actions = self.actor(states)
				q1 = tf.squeeze(self.critic_1(states, new_actions), 1)
				q2 = tf.squeeze(self.critic_2(states, new_actions), 1)
				critic_value = tf.minimum(q1, q2)
				actor_loss = -tf.reduce_mean(critic_value)
				actor_loss += tf.reduce_sum(self.actor.losses)
				actor_loss += 2.0 * tf.reduce_sum(tf.square(filtered_demo_actions - filtered_actor_actions))
			actor_gradient = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
			self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
			del actor_tape

			self.update_network_parameters()

			wandb.log({"Critic 1 Loss": critic_1_loss.numpy()})
			wandb.log({"Critic 2 Loss": critic_2_loss.numpy()})
			wandb.log({"Actor Loss": actor_loss.numpy()})


if __name__ == "__main__":

	env = gym.make("FetchPickAndPlace-v1", reward_type="dense")
	agent = Agent(env)

	score_history = []
	success_rate = []

	for demo in range(50):
		print("DEMO COLLECTION: ", demo)
		state = env.reset()
		dones = False

		while not dones:
			# env.render()

			observation = state["observation"]
			desired_goal = state["desired_goal"]

			action_choice = np.random.choice([1, 2], p=[0.9, 0.1])
			if action_choice == 1:
				action, o_mean, o_std, g_mean, g_std = demonstrations(env, observation, desired_goal)
				action = action + np.random.normal(scale=0.1)
			elif action_choice == 2:
				action = env.action_space.sample()

			next_state, _, dones, info = env.step(action)

			dist_block_goal = np.linalg.norm(next_state["achieved_goal"]-next_state["desired_goal"])
			dist_eef_block = np.linalg.norm(next_state["observation"][0:3]-next_state["achieved_goal"])
			dist_eef_goal = np.linalg.norm(next_state["observation"][0:3]-next_state["desired_goal"])

			if info["is_success"] == 0.0:
				done = False
				reward = -(3.0 * dist_block_goal) -(2.0 * dist_eef_block) -(1.0 * dist_eef_goal)

			if info["is_success"] == 1.0:
				done = True
				reward = 0.01 * (1/dist_eef_goal)

			o_clip = np.clip(state["observation"], -200, 200)
			g_clip = np.clip(state["desired_goal"], -200, 200)
			o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
			g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
			observation = np.concatenate([o_norm, g_norm])

			o_clip = np.clip(next_state["observation"], -200, 200)
			g_clip = np.clip(state["desired_goal"], -200, 200)
			o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
			g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
			next_observation = np.concatenate([o_norm, g_norm])

			agent.demo_remember(observation, action, reward, next_observation, done)
			agent.agent_remember(observation, action, reward, next_observation, done)

			state=next_state

	wandb.init(project="FetchPickAndPlace-v1", entity="Vidharth")

	for pre_train in range(1000):
		print("PRE TRAINING: ", pre_train)
		agent.demo_learn()

	for train in range(10000):
		start = time.time()
		state = env.reset()
		dones = False
		episodic_reward = 0

		while not dones:
			# env.render()

			o_clip = np.clip(state["observation"], -200, 200)
			g_clip = np.clip(state["desired_goal"], -200, 200)
			o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
			g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
			observation = np.concatenate([o_norm, g_norm])

			action_choice = np.random.choice([1, 2], p=[0.9, 0.1])
			if action_choice == 1:
				action = agent.train_choose_action(observation)
			elif action_choice == 2:
				action = env.action_space.sample()

			next_state, _, dones, info = env.step(action)

			dist_block_goal = np.linalg.norm(next_state["achieved_goal"]-next_state["desired_goal"])
			dist_eef_block = np.linalg.norm(next_state["observation"][0:3]-next_state["achieved_goal"])
			dist_eef_goal = np.linalg.norm(next_state["observation"][0:3]-next_state["desired_goal"])

			if info["is_success"] == 0.0:
				done = False
				reward = -(3.0 * dist_block_goal) -(2.0 * dist_eef_block) -(1.0 * dist_eef_goal)

			if info["is_success"] == 1.0:
				done = True
				reward = 0.01 * (1/dist_eef_goal)

			o_clip = np.clip(next_state["observation"], -200, 200)
			g_clip = np.clip(state["desired_goal"], -200, 200)
			o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
			g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
			next_observation = np.concatenate([o_norm, g_norm])

			agent.agent_remember(observation, action, reward, next_observation, done)

			state=next_state
			episodic_reward += reward

			train_choice = np.random.choice([1, 2], p=[0.8, 0.2])
			if train_choice == 1:
				agent.learn()
			elif train_choice == 2:
				agent.demo_learn()

		score_history.append(episodic_reward)
		wandb.log({"Average Reward": np.mean(score_history)})

		if train % 100 == 0:

			accuracy_1 = []
			for test1 in range(10):
				state = env.reset()
				done = False

				while not done:
					o_clip = np.clip(state["observation"], -200, 200)
					g_clip = np.clip(state["desired_goal"], -200, 200)
					o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
					g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
					observation = np.concatenate([o_norm, g_norm])

					action = agent.test_choose_action(observation)

					next_state, extrinsic_reward, done, info = env.step(action)

					state = next_state

				accuracy_1.append(info["is_success"])

			accuracy_2 = []
			for test2 in range(10):
				state = env.reset()
				done = False

				while not done:
					o_clip = np.clip(state["observation"], -200, 200)
					g_clip = np.clip(state["desired_goal"], -200, 200)
					o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
					g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
					observation = np.concatenate([o_norm, g_norm])

					action = agent.test_choose_action(observation)

					next_state, extrinsic_reward, done, info = env.step(action)

					state = next_state

				accuracy_2.append(info["is_success"])

			success_rate.append(np.mean([np.mean(accuracy_1), np.mean(accuracy_2)]))
			wandb.log({"Success Rate": success_rate[-1]})

			if success_rate[-1] >= max(success_rate):
				agent.save_networks()

			if success_rate[-1] > 0.9:
				agent.save_networks()
				exit()

		end = time.time()
		print("Episode: ", train, "Time: ", end-start)
