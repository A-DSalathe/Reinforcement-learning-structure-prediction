import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from environment import Molecule_Environment

# Assuming Molecule_Environment is defined as provided and properly imported
env = Molecule_Environment()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

state_size = np.prod(env.dimensions)  # Flattened state size
action_size = len(env.actions)
policy = Policy(state_size, action_size).to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

def get_flattened_state(state):
    return state.flatten()

def discount_rewards(rewards, gamma=0.99):
    discounted = []
    cumulative = 0
    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        discounted.insert(0, cumulative)
    return discounted

def reinforce(policy, optimizer, n_episodes=1000, max_t=2, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes + 1):
        print("starting episode: ",e)
        saved_log_probs = []
        rewards = []
        state = env.reset()
        flattened_state = get_flattened_state(state)
        for t in range(max_t):
            action_idx, log_prob = policy.act(flattened_state)
            saved_log_probs.append(log_prob)
            action = env.actions[action_idx]  # Convert action index to coordinates
            print("before step")
            print("action :", action)
            next_state, reward, done = env.step(action)
            print("after step")
            rewards.append(reward)
            state = next_state
            flattened_state = get_flattened_state(state)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        rewards_to_go = [sum([discounts[j] * rewards[j + t] for j in range(len(rewards) - t)]) for t in range(len(rewards))]

        policy_loss = []
        for log_prob, G in zip(saved_log_probs, rewards_to_go):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if e % print_every == 0:
            print(f'Episode {e}\tAverage Score: {np.mean(scores_deque):.2f}')

    return scores

env.print_spectra = True
scores = reinforce(policy, optimizer)
print("after reinforce")

# Use the trained policy to generate the molecule
state = env.reset()
flattened_state = get_flattened_state(state)
done = False

while not done:
    action_idx, _ = policy.act(flattened_state)
    action = env.actions[action_idx]  # Convert action index to coordinates
    state, _, done = env.step(action)
    flattened_state = get_flattened_state(state)

# Plotting the spectra
ref_spectra_y = env.ref_spectra[:, 1]
atom_pos = np.where(env.state == 1)
coords_atom = list(zip(*atom_pos))
spectra = spectra_from_arrays(positions=np.array(coords_atom) * env.resolution, chemical_symbols=env.chem_symbols, name=env.name, writing=False)
spectra_y = spectra[:, 1]

plt.figure(figsize=(10, 5))
plt.plot(env.ref_spectra[:, 0], ref_spectra_y, label='Reference Spectrum')
plt.plot(spectra[:, 0], spectra_y, label='Generated Spectrum', linestyle='--')
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Intensity')
plt.legend()
plt.show()