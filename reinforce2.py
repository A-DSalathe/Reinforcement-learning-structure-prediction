from environment import Molecule_Environment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from spectra import spectra_from_arrays
import os
import os.path as op
import shutil
from save_plot import *

script_dir = op.dirname(op.realpath(__file__))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

    def act(self, state_features, mode='explore', epsilon=0.1):
        state_features = torch.from_numpy(state_features).float().unsqueeze(0).to(device)
        probs = self.forward(state_features).squeeze(0)
        m = Categorical(probs)

        greedy_action = torch.argmax(probs).item()

        if mode == 'greedy':
            action = greedy_action
        else:
            # Define the probability distribution
            random_prob = 1 / 3
            greedy_prob = 2 / 3
            neighbor_prob = greedy_prob / 2
            actual_greedy_prob = greedy_prob - neighbor_prob

            # Decision logic
            p = np.random.rand()
            if p < random_prob:
                # Completely random action
                action = m.sample().item()
            elif p < (random_prob + actual_greedy_prob):
                # Greedy action
                action = greedy_action
            else:
                # Explore one of the 4 neighbors
                neighbors = self.get_neighbors(greedy_action)
                action = np.random.choice(neighbors)

        log_prob = m.log_prob(torch.tensor(action))
        return action, log_prob

    def get_neighbors(self, action_idx):
        # Convert the action index to its coordinate
        dims = (11, 11, 11)  # Assuming the same dimensions as the environment
        x, y, z = np.unravel_index(action_idx, dims)

        neighbors = []
        # Generate the possible neighbor coordinates
        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]:
                neighbors.append(np.ravel_multi_index((nx, ny, nz), dims))

        return neighbors


def reinforce(policy, optimizer, env, n_episodes=1000, max_t=10, gamma=0.99, print_every=2, eval_every=10,
              epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
    scores_deque = deque(maxlen=100)
    scores = []
    eval_losses = []
    eval_rewards = []
    epsilon = epsilon_start

    for e in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        state_features = env.get_state_features()
        for t in range(max_t):
            action_idx, log_prob = policy.act(state_features, mode='explore', epsilon=epsilon)
            saved_log_probs.append(log_prob)
            action = env.actions[action_idx]  # Convert action index to coordinates
            next_state, reward, done = env.step(action)
            rewards.append(reward)
            state = next_state
            state_features = env.get_state_features()
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards))]
        rewards_to_go = [sum([discounts[j] * rewards[j + t] for j in range(len(rewards) - t)]) for t in
                         range(len(rewards))]

        policy_loss = []
        for log_prob, G in zip(saved_log_probs, rewards_to_go):
            policy_loss.append(-log_prob * torch.tensor(G, dtype=torch.float32))

        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.01)

        optimizer.step()

        epsilon = max(epsilon_end, epsilon_decay * epsilon)  # Decay epsilon

        if e % print_every == 0:
            print(f'Episode {e}\t Score: {sum(rewards)}')

        if e % eval_every == 0:
            eval_reward, eval_loss = compute_greedy_reward_and_loss(env, policy)
            eval_rewards.append(eval_reward)
            eval_losses.append(eval_loss)
            print(f'Episode {e}\t Evaluation Loss: {eval_loss}\t Greedy Reward: {eval_reward}')

    print("Final Evaluation Losses:", eval_losses)
    print("Final Evaluation Rewards:", eval_rewards)
    plot_eval_loss_and_rewards(eval_losses, eval_rewards, 'eval_loss_and_rewards_' + str(number_of_atoms) + '_atoms',
                               display=True)
    print(scores)
    return scores, eval_losses, eval_rewards


def compute_greedy_reward_and_loss(env, policy):
    state = env.reset()
    state_features = env.get_state_features()
    done = False
    total_reward = 0

    while not done:
        action_idx, _ = policy.act(state_features, mode='greedy')
        action = env.actions[action_idx]  # Convert action index to coordinates
        state, reward, done = env.step(action)
        total_reward += reward
        state_features = env.get_state_features()

    eval_loss = -env.diff_spectra()
    return total_reward, eval_loss


if __name__ == "__main__":
    number_of_atoms = 2
    env = Molecule_Environment(n_atoms=number_of_atoms, chemical_symbols=["B"], dimensions=(11, 11, 11),
                               resolution=np.array([0.2, 0.2, 0.2]),
                               ref_spectra_path=op.join(script_dir, op.join('references', 'reference_1_B.dat')),
                               print_spectra=0)
    input_dim = number_of_atoms * (number_of_atoms - 1) // 2  # Maximum number of pairwise distances
    action_size = len(env.actions)
    policy = PolicyNetwork(input_dim, 16, action_size).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=0.0001, weight_decay=0.01)
    dir_path = "ir"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    scores, eval_losses, eval_rewards = reinforce(policy, optimizer, env, n_episodes=500, eval_every=10)
    save_weights(policy, 'weight_' + str(number_of_atoms) + '_atoms')

    state = env.reset()
    state_features = env.get_state_features()
    done = False

    while not done:
        action_idx, _ = policy.act(state_features, mode='greedy')
        action = env.actions[action_idx]
        state, _, done = env.step(action)
        state_features = env.get_state_features()

    ref_spectra_y = env.ref_spectra[:, 1]
    atom_pos = np.where(env.state != 0)
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

    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[0, 0],
                       title='front_view_' + str(number_of_atoms) + '_atoms', display=False)
    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[0, 90],
                       title='side_view_' + str(number_of_atoms) + '_atoms', display=False)
    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[90, 0],
                       title='top_view_' + str(number_of_atoms) + '_atoms', display=False)
    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[30, 30],
                       title='3d_structure_' + str(number_of_atoms) + '_atoms', display=True)

    state_features_test = env.get_state_features()
    action, log_prob = policy.act(state_features_test, mode='greedy')
    print(log_prob)