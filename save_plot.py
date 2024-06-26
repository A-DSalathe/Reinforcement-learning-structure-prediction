import numpy as np
from matplotlib import pyplot as plt
import os
import os.path as op
from mpl_toolkits.mplot3d import Axes3D
import torch
script_dir = op.dirname(op.realpath(__file__))

def save_array(array,title):
    folder_name = 'numpy_array_folder'
    folder_path = op.join(script_dir,folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = op.join(folder_path,title+'.npy')
    np.save(file_path, array)

def plot_scores(scores, title, display=False):
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
    folder_score = 'score'
    folder_path = op.join(script_dir,folder_score)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = op.join(folder_path,title+'.png')
    plt.savefig(file_path)
    if display:
        plt.show()

def plot_sphere(ax, center, radius, color='r'):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.6)
def calculate_centroid(positions):
    return np.mean(positions, axis=0)

def center_molecule(positions, grid_dimensions, resolution):
    centroid = calculate_centroid(positions)
    grid_center = np.array(grid_dimensions) * resolution / 2.0
    shift_vector = grid_center - centroid
    shifted_positions = positions + shift_vector
    return shifted_positions

def plot_and_save_view(positions, resolution, grid_dimensions, view_angle, title, display=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Center the molecule
    centered_positions = center_molecule(np.array(positions), grid_dimensions, resolution)

    # Plot spheres at atom positions
    for pos in centered_positions:
        plot_sphere(ax, pos, radius=0.1, color='r')  # Adjust the radius and color as needed


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

    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    folder_3d_plot = '3d_plot'
    folder_path = op.join(script_dir, folder_3d_plot)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = op.join(folder_path, title + '.png')
    plt.savefig(file_path)

    if display:
        plt.show()
    else:
        plt.close(fig)  # Close the figure to free up memory


def plot_spectra(arrays, title, display=False):
    plt.figure(figsize=(10, 5))
    ref_frequency = arrays[0]
    ref_spectra_y = arrays[1]
    frequency = arrays[2]
    spectra_y = arrays[3]
    plt.plot(ref_frequency, ref_spectra_y, label='Reference Spectrum')
    plt.plot(frequency, spectra_y, label='Generated Spectrum', linestyle='--')
    plt.xlabel('Wavenumber (cm^-1)')
    plt.ylabel('Intensity')
    plt.legend()
    folder_spectra = 'spectra'
    folder_path = op.join(script_dir,folder_spectra)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = op.join(folder_path,title+'.png')
    plt.savefig(file_path)
    if display:
        plt.show()
        
def save_weights(policy,name):
    folder_weight = 'weight'
    folder_path = op.join(script_dir,folder_weight)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = op.join(folder_path,name+'.pth')
    torch.save(policy.state_dict(), file_path)

def plot_eval_loss_and_rewards(eval_losses, eval_rewards, title, display=False):
    plt.figure(figsize=(10, 5))
    intervals = np.arange(0, len(eval_losses) * 10, 10)  # Assuming eval_every is 10
    plt.plot(intervals, eval_losses, label='Evaluation Loss')
    plt.plot(intervals, eval_rewards, label='Greedy Reward')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Evaluation Loss and Greedy Reward over Time')
    plt.legend()
    folder_eval_loss = 'eval_loss_and_rewards'
    folder_path = op.join(script_dir, folder_eval_loss)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = op.join(folder_path, title + '.png')
    plt.savefig(file_path)
    if display:
        plt.show()
def read_npy(path):
    data = np.genfromtxt(path, dtype=float, skip_header=2, usecols=(1, 2, 3))
    return data
if __name__ == "__main__":
    # test_array = np.array([[1,2,4],[1,2,3]])
    # test_name = 'name'
    # save_array(array=test_array,title=test_name)
    # folder_name = 'numpy_array_folder'
    # folder_path = op.join(script_dir,folder_name)
    # file_path = op.join(folder_path,test_name+'.npy')
    # loaded_array = np.load(file_path)
    # print(loaded_array)
    array = read_npy(op.join(script_dir,op.join('references','reference_1_B.xyz')))
    plot_and_save_view(array, [0.2, 0.2, 0.2], [11, 11, 11], [30, 30], 'ref', display=True)
