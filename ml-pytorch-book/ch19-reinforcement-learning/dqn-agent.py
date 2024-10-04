# Users a Neural Network to approximate the Q function and select actions
# the agent will take in the environment to optimize its rewards.

import random
import gym
from gymnasium.wrappers import RecordVideo
from collections import deque
from collections import namedtuple
from IPython import display

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNAgent:
    def __init__(
            self, env, discount_factor=0.95,
            epsilon_greedy=1.0, epsilon_min=0.01,
            epsilon_decay=0.995, learning_rate=1e-3,
            max_memory_size=2000):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self._build_nn_model()

    def _build_nn_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def remember(self, transition):
        self.memory.append(transition)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Gives a change to a random action to increase exploration
            return np.random.choice(self.action_size)
        with torch.no_grad():
            # Uses the NN to predict the Q values of the state
            # to select the best action to take. This focus on exploitation.
            q_values = self.model(torch.tensor(state, dtype=torch.float32))[0]
        return torch.argmax(q_values).item()  # returns action

    def _learn(self, batch_samples):
        batch_states, batch_targets = [], []
        # Uses a batch of samples to train the model to avoid having
        # a single action to affect other actions. This is a strategy
        # to improve q-learning training.
        for transition in batch_samples:
            s, a, r, next_s, done = transition
            with torch.no_grad():
                if done:
                    target = r
                else:
                    # Use the model to predict the Q values of the next state
                    pred = self.model(torch.tensor(next_s, dtype=torch.float32))[0]
                    target = r + self.gamma * pred.max()
            # Use the model to predict the Q values of the current state
            target_all = self.model(torch.tensor(s, dtype=torch.float32))[0]
            # Use the reward when done or from the next state as target to train the model
            target_all[a] = target
            batch_states.append(s.flatten())
            batch_targets.append(target_all)

        self.optimizer.zero_grad()
        pred = self.model(torch.tensor(batch_states, dtype=torch.float32))
        loss = self.loss_fn(pred, torch.stack(batch_targets))
        loss.backward()
        self.optimizer.step()

        self._adjust_epsilon()
        return loss.item()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return self._learn(samples)


def plot_learning_history(history):
    fig = plt.figure(1, figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1)
    episodes = np.arange(len(history)) + 1
    plt.plot(episodes, history, lw=4, marker='o', markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('Total rewards', size=20)
    plt.show()


# General settings
EPISODES = 200
batch_size = 32
init_replay_memory_size = 500

if __name__ == '__main__':
    # Uses the CartPole environment to test the DQNAgent
    trigger = lambda t: t % 10 == 0
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=trigger, disable_logger=True)

    agent = DQNAgent(env)
    state = env.reset()
    state = np.reshape(state[0], [1, agent.state_size])

    # Filling up the replay-memory
    # This is just so the Neural Network can learn from a batch of already filled samples
    for i in range(init_replay_memory_size):
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        agent.remember(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = np.reshape(state[0], [1, agent.state_size])
        else:
            state = next_state

    # Train the agent for a number of episodes
    total_rewards, losses = [], []
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state[0], [1, agent.state_size])
        for i in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(Transition(state, action, reward, next_state, done))
            state = next_state
            if e % 10 == 0:
                env.render()
            if done:
                total_rewards.append(i)
                print(f'Episode: {e}/{EPISODES}, Total reward: {i}')
                break
            loss = agent.replay(batch_size)
            losses.append(loss)

    plot_learning_history(total_rewards)
