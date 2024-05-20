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


def plot_scores(scores):
    # Convert scores to a numpy array for easier manipulation
    scores_array = np.array(scores)

    # Identify and filter out extreme values
    filtered_scores = np.clip(scores_array, a_min=-10, a_max=1)

    plt.figure(figsize=(10, 5))
    plt.plot(filtered_scores, label='Score per Episode')
    plt.ylim(
        [np.min(filtered_scores) - 1, np.max(filtered_scores) + 1])  # Adjust y-axis limits for better visualization
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Evolution of the Score over Episodes')
    plt.legend()
    plt.show()


def reinforce(policy, optimizer, n_episodes=100, max_t=10, gamma=1.0, print_every=2, epsilon_start=1.0, epsilon_end=0.1,
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
            #might be adding two times the reward
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

    plot_scores(scores)  # Plot the scores at the end of training
    print(scores)
    return scores


def calculate_distance(point1, point2):
    distance = np.linalg.norm(np.array(point2) - np.array(point1))
    return distance


############################
# 3d ploting
def plot_sphere(ax, center, radius, color='r'):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.6)


def plot_3d_structure(positions, resolution, grid_dimensions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for pos in positions:
        plot_sphere(ax, pos, radius=0.1)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([0, grid_dimensions[0] * resolution[0]])
    ax.set_ylim([0, grid_dimensions[1] * resolution[1]])
    ax.set_zlim([0, grid_dimensions[2] * resolution[2]])

    ax.set_box_aspect([grid_dimensions[0] * resolution[0],
                       grid_dimensions[1] * resolution[1],
                       grid_dimensions[2] * resolution[2]])

    plt.show()

if __name__ == "__main__":
    # Assuming Molecule_Environment is defined as provided and properly imported
    number_of_atoms = 7
    env = Molecule_Environment(n_atoms = number_of_atoms, chemical_symbols = ["B"], dimensions = (21,21,21), resolution=np.array([0.1,0.1,0.1]), ref_spectra_path = op.join(script_dir,op.join('references','reference_custom_1.dat')), print_spectra=0)
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
    scores = reinforce(policy, optimizer, n_episodes=100)

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

    plt.figure(figsize=(10, 5))
    plt.plot(env.ref_spectra[:, 0], ref_spectra_y, label='Reference Spectrum')
    plt.plot(spectra[:, 0], spectra_y, label='Generated Spectrum', linestyle='--')
    plt.xlabel('Wavenumber (cm^-1)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()


    resolution = env.resolution
    grid_dimensions = env.dimensions
    print(grid_dimensions)
    plot_3d_structure(positions, resolution, grid_dimensions)

    flattened_state_test = get_flattened_state(env.reset())
    action, log_prob = policy.act(flattened_state_test)
    print(log_prob)
    #there is clearly a big problem with the scores as it seems it doesn't go down