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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil
import math
from save_plot import *

script_dir = op.dirname(op.realpath(__file__))



# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy3DCNN(nn.Module):
    def __init__(self, input_shape, action_size, hidden_size=64):
        super(Policy3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_shape[0] * input_shape[1] * input_shape[2], hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state, epsilon=0.1):
        state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(device)
        probs = self.forward(state).cpu().detach().numpy().squeeze()

        if np.random.rand() < epsilon:
            # Exploration: randomly select an action
            action = np.random.choice(len(probs))
            log_prob = np.log(probs[action])
        else:
            # Exploitation: select the action with the highest probability
            action = np.argmax(probs)
            log_prob = np.log(probs[action])

        return action, log_prob


def get_3d_state(state):
    return state  # Now state is already the 3D grid

def reinforce(policy, optimizer, env, n_episodes=100, max_t=10, gamma=1.0, print_every=2, eval_every=10, epsilon_start=1.0, epsilon_end=1.0, epsilon_decay=0.995):
    scores_deque = deque(maxlen=100)
    scores = []
    eval_losses = []
    eval_rewards = []
    epsilon = epsilon_start

    for e in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        flattened_state = state
        for t in range(max_t):
            action_idx, log_prob = policy.act(flattened_state, epsilon)
            saved_log_probs.append(log_prob)
            action = env.actions[action_idx]  # Convert action index to coordinates
            next_state, reward, done = env.step(action)
            rewards.append(reward)
            state = next_state
            flattened_state = state
            if done:
                break

        n_points = len(np.where(state == 1)[0])
        if done and n_points != env.n_atoms:
            rewards.append(-10)

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        rewards_to_go = [sum([discounts[j] * rewards[j + t] for j in range(len(rewards) - t)]) for t in
                         range(len(rewards))]

        policy_loss = []
        for log_prob, G in zip(saved_log_probs, rewards_to_go):
            policy_loss.append(-log_prob * torch.tensor(G, dtype=torch.float32, requires_grad=True))

        if policy_loss:
            policy_loss = torch.stack(policy_loss).sum()
        else:
            policy_loss = torch.tensor(0.0, requires_grad=True)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        epsilon = max(epsilon_end, epsilon_decay * epsilon)  # Decay epsilon

        if e % print_every == 0:
            print(f'Episode {e}\t Score: {sum(rewards)}')

        # Evaluate loss and reward at regular intervals using greedy policy
        if e % eval_every == 0:
            eval_reward, eval_loss = compute_greedy_reward_and_loss(env, policy)
            eval_rewards.append(eval_reward)
            eval_losses.append(eval_loss)
            print(f'Episode {e}\t Evaluation Loss: {eval_loss}\t Greedy Reward: {eval_reward}')

    plot_eval_loss_and_rewards(eval_losses, eval_rewards, 'eval_loss_and_rewards_' + str(env.n_atoms) + '_atoms', display=False)
    print(scores)
    return scores, eval_losses, eval_rewards

def compute_greedy_reward_and_loss(env, policy):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_idx, _ = policy.act(state, epsilon=0.0)
        action = env.actions[action_idx]  # Convert action index to coordinates
        state, reward, done = env.step(action)
        total_reward += reward

    # Compute the loss using the environment's diff_spectra function
    eval_loss = - env.diff_spectra()
    return total_reward, eval_loss

if __name__ == "__main__":
    # Assuming Molecule_Environment is defined as provided and properly imported
    number_of_atoms = 2
    env = Molecule_Environment(n_atoms=number_of_atoms, chemical_symbols=["B"], dimensions=(5, 5, 5), resolution=np.array([0.25, 0.25, 0.25]), ref_spectra_path=op.join(script_dir, op.join('references', 'reference_1_B.dat')), print_spectra=0)
    input_shape = env.dimensions
    state_size = 1  # Not used for 3D CNN
    action_size = len(env.actions)
    policy = Policy3DCNN(input_shape, action_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    dir_path = "ir"  # Change this to an absolute path if needed, e.g., r"C:\path\to\ir"
    if os.path.exists(dir_path):
        print(f"Directory {dir_path} exists, attempting to remove it.")
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
            print(f"Directory {dir_path} removed successfully.")
        except Exception as e:
            print(f"An error occurred while trying to remove the directory: {e}")
    else:
        print(f"Directory {dir_path} does not exist.")
    scores, eval_losses, eval_rewards = reinforce(policy, optimizer, env, n_episodes=100, eval_every=10)
    save_weights(policy, 'weight_' + str(number_of_atoms) + '_atoms')

    # Use the trained policy to generate the molecule
    state = env.reset()
    done = False

    while not done:
        action_idx, _ = policy.act(state, epsilon=0.0)
        action = env.actions[action_idx]  # Convert action index to coordinates
        state, _, done = env.step(action)

    # Plotting the spectra
    ref_spectra_y = env.ref_spectra[:, 1]
    atom_pos = np.where(env.state == 1)
    coords_atom = list(zip(*atom_pos))
    positions = np.array(coords_atom) * env.resolution
    spectra = spectra_from_arrays(positions=positions, chemical_symbols=env.chem_symbols, name=env.name, writing=False)
    print(positions)
    spectra_y = spectra[:, 1]
    np.where(env.state == 1)
    print(policy)

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
    print(grid_dimensions)
    save_array(positions, 'pos_' + str(number_of_atoms) + '_atoms')
    save_array(resolution, 'res_' + str(number_of_atoms) + '_atoms')
    save_array(grid_dimensions, 'grid_dim_' + str(number_of_atoms) + '_atoms')

    # Save the 3D plot with multiple views
    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[0, 0],
                       title='front_view_' + str(number_of_atoms) + '_atoms', display=False)
    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[0, 90],
                       title='side_view_' + str(number_of_atoms) + '_atoms', display=False)
    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[90, 0],
                       title='top_view_' + str(number_of_atoms) + '_atoms', display=False)
    plot_and_save_view(positions, resolution, grid_dimensions, view_angle=[30, 30],
                       title='3d_structure_' + str(number_of_atoms) + '_atoms', display=True)

    flattened_state_test = get_3d_state(env.reset())
    action, log_prob = policy.act(flattened_state_test)
    print(log_prob)

