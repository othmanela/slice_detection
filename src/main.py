import os
import numpy as np
import pandas as pd
from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F

from agent import Agent
from dqn import DQN
from penv import PatientEnvManager
from qvalues import QValues
from replay_mem import ReplayMemory
from strategy import EpsilonGreedyStrategy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN()
dqn.to(device)

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    return (t1, t2, t3, t4)


batch_size = 48
gamma = 0.99
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 17000
lr = 0.001
num_episodes = 100000

em = PatientEnvManager(device, csv_file_train)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()

    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            episode_durations.append(timestep)
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

em.close()
em_val.close()

MODEL_PATH = ''
torch.save(target_net.state_dict(), MODEL_PATH)
