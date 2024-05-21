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


    
if __name__ == "__main__":
    test_array = np.array([[1,2,4],[1,2,3]])
    test_name = 'name'
    save_array(array=test_array,title=test_name)
    folder_name = 'numpy_array_folder'
    folder_path = op.join(script_dir,folder_name)
    file_path = op.join(folder_path,test_name+'.npy')
    loaded_array = np.load(file_path)
    print(loaded_array)