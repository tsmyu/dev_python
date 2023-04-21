import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(observation_size, 16)
        self.l2 = nn.Linear(16, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)

        return x


class Agent:
    def __init__(self, obvservation_size):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(obvservation_size, self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), self.lr)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs_detached = probs.detach().numpy()[0]

        action = np.random.choice(len(probs_detached), p=probs_detached)


        return action, probs_detached[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        self.pi.cleargrads()
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G

        for reward, prob in self.memory:
            loss += -F.log(prob) + G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


episodes = 3000
env = gym.make("CartPole-v1")
state = torch.from_numpy(env.reset()[0]).clone()
agent = Agent(len(state))
reward_history = []
for episode in range(episodes):
    state = torch.from_numpy(env.reset()[0]).clone()
    done = False
    total_reward = 0

    while not terminated:
        print("state", state)
        action, prob = agent.get_action(state)
        print(env.step(action))
        observation, reward, terminated, truncated, info = env.step(action)
        agent.add(reward, prob)
        state = observation
        total_reward += reward

    agent.update()
    reward_history.append(total_reward)
