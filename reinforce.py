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


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state, epsilon=0.1):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
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



def get_flattened_state(state):
    return state.flatten()


def discount_rewards(rewards, gamma=0.99):
    discounted = []
    cumulative = 0
    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        discounted.insert(0, cumulative)
    return discounted





def reinforce(policy, optimizer, env, n_episodes=100, max_t=10, gamma=1.0, print_every=2, epsilon_start=1.0, epsilon_end=0.1,
              epsilon_decay=0.995):
    scores_deque = deque(maxlen=100)
    scores = []
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
            print(f'Episode {e}\t Score: {sum(rewards):.2f}')

    plot_scores(scores,'score_'+str(number_of_atoms)+'_atoms',display=False)  # Plot the scores at the end of training
    print(scores)
    return scores


def calculate_distance(point1, point2):
    distance = np.linalg.norm(np.array(point2) - np.array(point1))
    return distance


############################
# 3d ploting


if __name__ == "__main__":
    # Assuming Molecule_Environment is defined as provided and properly imported
    number_of_atoms = 2
    env = Molecule_Environment(n_atoms = number_of_atoms, chemical_symbols = ["B"], dimensions = (21,21,21), resolution=np.array([0.1,0.1,0.1]), ref_spectra_path = op.join(script_dir,op.join('references','reference_1_B.dat')), print_spectra=0)
    flatten_dimensions = np.prod(env.dimensions)
    # state_size = math.comb(flatten_dimensions, number_of_atoms-1)  # Flattened state size
    print(flatten_dimensions)
    state_size = flatten_dimensions
    # state_size = 10**6
    action_size = len(env.actions)
    policy = Policy(state_size, action_size).to(device)
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
    scores = reinforce(policy, optimizer, env, n_episodes=10)

    # Use the trained policy to generate the molecule
    state = env.reset()
    flattened_state = get_flattened_state(state)
    done = False

    while not done:
        action_idx, _ = policy.act(flattened_state,epsilon=0.0)
        action = env.actions[action_idx]  # Convert action index to coordinates
        state, _, done = env.step(action)
        flattened_state = get_flattened_state(state)

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
    #################################################
    # test function
    point1 = [-0.475, -0.475, 0.0]
    point2 = [0.475, 0.475, 0.0]

    print(calculate_distance(point1, point2))
    print(calculate_distance(positions[0], positions[1]))

    ##################

    arrays = []
    arrays.append(env.ref_spectra[:, 0])
    arrays.append(ref_spectra_y)
    arrays.append(spectra[:, 0])
    arrays.append(spectra_y)
    name = 'spectra_'+str(number_of_atoms)+'_atoms'
    plot_spectra(arrays=arrays,title=name,display=False)
    # plt.figure(figsize=(10, 5))
    # plt.plot(env.ref_spectra[:, 0], ref_spectra_y, label='Reference Spectrum')
    # plt.plot(spectra[:, 0], spectra_y, label='Generated Spectrum', linestyle='--')
    # plt.xlabel('Wavenumber (cm^-1)')
    # plt.ylabel('Intensity')
    # plt.legend()
    # plt.show()


    resolution = env.resolution
    grid_dimensions = env.dimensions
    print(grid_dimensions)
    save_array(positions,'pos_'+str(number_of_atoms)+'_atoms')
    save_array(resolution,'res_'+str(number_of_atoms)+'_atoms')
    save_array(grid_dimensions,'grid_dim_'+str(number_of_atoms)+'_atoms')
    plot_3d_structure(positions, resolution, grid_dimensions)

    flattened_state_test = get_flattened_state(env.reset())
    action, log_prob = policy.act(flattened_state_test)
    print(log_prob)

