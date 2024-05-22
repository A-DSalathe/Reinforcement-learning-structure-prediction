from environment import Molecule_Environment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from spectra import spectra_from_arrays
import os
import os.path as op
import shutil
import math
from save_plot import *

script_dir = op.dirname(op.realpath(__file__))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        #self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state, epsilon=0.1):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).squeeze(0)
        m = Categorical(probs)
        if np.random.rand() < epsilon:
            action = m.sample()
        else:
            action = torch.argmax(probs)
        log_prob = m.log_prob(action)
        return action.item(), log_prob

def get_flattened_state(state):
    return state.flatten()

def discount_rewards(rewards, gamma=0.99):
    discounted = []
    cumulative = 0
    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        discounted.insert(0, cumulative)
    return discounted

def reinforce(policy, optimizer, env, n_episodes=100, max_t=10, gamma=0.99, print_every=2, eval_every=10, epsilon_start=1.0, epsilon_end=0.9, epsilon_decay=0.995):
    scores_deque = deque(maxlen=100)
    scores = []
    eval_losses = []
    eval_rewards = []
    epsilon = epsilon_start

    for e in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        flattened_state = get_flattened_state(state)
        for t in range(max_t):
            action_idx, log_prob = policy.act(flattened_state, epsilon)
            saved_log_probs.append(log_prob)
            action = env.actions[action_idx]  # Convert action index to coordinates
            next_state, reward, done = env.step(action)
            rewards.append(reward)
            state = next_state
            flattened_state = get_flattened_state(state)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards))]
        rewards_to_go = [sum([discounts[j] * rewards[j + t] for j in range(len(rewards) - t)]) for t in range(len(rewards))]

        policy_loss = []
        for log_prob, G in zip(saved_log_probs, rewards_to_go):
            policy_loss.append(-log_prob * torch.tensor(G, dtype=torch.float32))

        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()

        # Debugging: Print gradients
        for name, param in policy.named_parameters():
            if param.grad is not None:
                print(f'Gradients for {name}: {param.grad}')
            else:
                print(f'No gradients for {name}')

        optimizer.step()

        epsilon = max(epsilon_end, epsilon_decay * epsilon)  # Decay epsilon

        if e % print_every == 0:
            print(f'Episode {e}\t Score: {sum(rewards):.2f}')

        if e % eval_every == 0:
            eval_reward, eval_loss = compute_greedy_reward_and_loss(env, policy)
            eval_rewards.append(eval_reward)
            eval_losses.append(eval_loss)
            print(f'Episode {e}\t Evaluation Loss: {eval_loss}\t Greedy Reward: {eval_reward}')

    print("Final Evaluation Losses:", eval_losses)
    print("Final Evaluation Rewards:", eval_rewards)
    plot_eval_loss_and_rewards(eval_losses, eval_rewards, 'eval_loss_and_rewards_' + str(number_of_atoms) + '_atoms', display=True)
    print(scores)
    return scores, eval_losses, eval_rewards

def compute_greedy_reward_and_loss(env, policy):
    state = env.reset()
    flattened_state = get_flattened_state(state)
    done = False
    total_reward = 0

    while not done:
        action_idx, _ = policy.act(flattened_state, epsilon=0.0)
        action = env.actions[action_idx]  # Convert action index to coordinates
        state, reward, done = env.step(action)
        total_reward += reward
        flattened_state = get_flattened_state(state)

    eval_loss = -env.diff_spectra()
    return total_reward, eval_loss

if __name__ == "__main__":
    number_of_atoms = 2
    env = Molecule_Environment(n_atoms=number_of_atoms, chemical_symbols=["B"], dimensions=(11, 11, 11), resolution=np.array([0.1, 0.1, 0.1]), ref_spectra_path=op.join(script_dir, op.join('references', 'reference_1_B.dat')), print_spectra=0)
    flatten_dimensions = np.prod(env.dimensions)
    state_size = flatten_dimensions
    action_size = len(env.actions)
    policy = Policy(state_size, action_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.005)
    dir_path = "ir"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    scores = reinforce(policy, optimizer, env, n_episodes=100, eval_every=1)
    save_weights(policy, 'weight_' + str(number_of_atoms) + '_atoms')

    state = env.reset()
    flattened_state = get_flattened_state(state)
    done = False

    while not done:
        action_idx, _ = policy.act(flattened_state, epsilon=0.0)
        action = env.actions[action_idx]
        state, _, done = env.step(action)
        flattened_state = get_flattened_state(state)

    ref_spectra_y = env.ref_spectra[:, 1]
    atom_pos = np.where(env.state == 1)
    coords_atom = list(zip(*atom_pos))
    positions = np.array(coords_atom) * env.resolution
    spectra = spectra_from_arrays(positions=positions, chemical_symbols=env.chem_symbols, name=env.name, writing=False)
    spectra_y = spectra[:, 1]

    arrays = []
    arrays.append(env.ref_spectra[:, 0])
    arrays.append(ref_spectra_y)
    arrays.append(spectra[:, 0])
    arrays.append(spectra_y)
    name = 'spectra_' + str(number_of_atoms) + '_atoms'
    save_array(spectra, title=name)
    plot_spectra(arrays=arrays, title=name, display=False)

    resolution = env.resolution
    grid_dimensions = env.dimensions
    save_array(positions, 'pos_' + str(number_of_atoms) + '_atoms')
    save_array(resolution, 'res_' + str(number_of_atoms) + '_atoms')
    save_array(grid_dimensions, 'grid_dim_' + str(number_of_atoms) + '_atoms')

    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[0, 0], title='front_view_' + str(number_of_atoms) + '_atoms', display=False)
    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[0, 90], title='side_view_' + str(number_of_atoms) + '_atoms', display=False)
    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[90, 0], title='top_view_' + str(number_of_atoms) + '_atoms', display=False)
    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[30, 30], title='3d_structure_' + str(number_of_atoms) + '_atoms', display=True)

    flattened_state_test = get_flattened_state(env.reset())
    action, log_prob = policy.act(flattened_state_test)
    print(log_prob)
