import numpy as np
from matplotlib import pyplot as plt
import os
import os.path as op

script_dir = op.dirname(op.realpath(__file__))

def save_array(array,name):
    folder_name = 'numpy_array_folder'
    folder_path = op.join(script_dir,folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = op.join(folder_path,name+'.npy')
    np.save(file_path, array)





if __name__ == "__main__":
    test_array = np.array([[1,2,3],[1,2,3]])
    test_name = 'name'
    save_array(array=test_array,name=test_name)
    folder_name = 'numpy_array_folder'
    folder_path = op.join(script_dir,folder_name)
    file_path = op.join(folder_path,test_name+'.npy')
    loaded_array = np.load(file_path)
    print(loaded_array)