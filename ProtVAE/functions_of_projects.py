import json
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
#os.system("!pip install py3Dmol")

# Read JSON file
def read_json_file(file_path):
  data = []
  with open(file_path, 'r') as file:
    for line in file:
      obj = json.loads(line)
      data.append(obj)
  return data

# Read JSONL file
def read_jsonl_file(file_path):
  data = []
  with open(file_path, 'r') as file:
    for line in file:
      obj = json.loads(line)
      data.append(obj)
  return data

# Analyze JSON data
def analyze_json_data(data, limit):
  for i, obj in enumerate(data[:limit]):
    name = obj
    # Operations are performed here
    # For example, printing only the names to the screen
    print(f"JSON Data {i+1}: Name: {name}")

# Analyze JSONL data
def analyze_jsonl_data(data, limit):
  for i, obj in enumerate(data[:limit]):
    name = obj['name']
    coordinates = obj['coords']
    sequence = obj['seq']
      
    # Operations are performed here
    # For example, printing names, coordinates, and sequences to the screen
    print(f"JSONL Data {i+1}: Name: {name} coordinates: {coordinates} sequence: {sequence}")

def plot_data_x(length_arr, plt_feature, plt_header):
  arr = np.arange(length_arr)


  plt.figure(figsize=(12, 8), dpi=100)
  plt.plot(arr, plt_feature, linestyle='-', linewidth=1)


  plt.xlabel('value')
  plt.ylabel('length_'+plt_header)
  plt.title('Data vs Length of '+plt_header+'  Max Size: '+str(max(plt_feature))+" Min Size: "+str(min(plt_feature)) + " Mean Size: "+str(np.mean(plt_feature)))


  plt.show()
  plt.savefig("/content/drive/MyDrive/CENG796/figures/data" + plt_header+ ".png")

def plot_data(length_arr, plt_feature, plt_header):
  arr = np.arange(length_arr)

  length_feature = calc_len_array(plt_feature)

  plt.figure(figsize=(12, 8), dpi=100)
  plt.plot(arr, length_feature, linestyle='-', linewidth=1)

  plt.xlabel('value')
  plt.ylabel(plt_header)
  plt.title('Data vs Length of '+plt_header+' Max Size: '+str(max(length_feature))+" Min Size: "+str(min(length_feature)) + " Mean Size: "+str(np.mean(length_feature)))

  plt.show()
  plt.savefig("/content/drive/MyDrive/CENG796/figures/data" + plt_header+ ".png")
def calc_len_array(array):
  ij = 0
  length_array = []
  while ij < len(array):
    length_array.append(len(array[ij]))
    ij = ij + 1
  return length_array

def normalize_data_len_10_50(data):
   return int((len(data) - 40) * (30 - 10) / (500 - 40) + 10)

def find_index_in_jsonl(element, jsonl_data):
  j = 0
  while j < len(jsonl_data):
    if (jsonl_data[j]["name"] == element):
      return j
    else:
      j = j + 1
  print("Error!!!!!!")

def normalize_sequence_func(sequence):
    amino_acid_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 21, 'B': 22, 'Z': 23, 'J': 24, 'U': 25, 'O': 26}
    normalized_sequence = [amino_acid_dict[aa] for aa in sequence]
    return normalized_sequence

def three_d_array_control(value):
  if math.isnan(value):
      return 0
  else:
      return value

def return_tensor(in_val, out_val, sequence_input, reduct_coords_N_y,reduct_coords_N_z,reduct_coords_CA_x,reduct_coords_CA_y,reduct_coords_CA_z,reduct_coords_C_x,reduct_coords_C_y,reduct_coords_C_z,reduct_coords_O_x,reduct_coords_O_y,reduct_coords_O_z):
    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []
    array_5 = []
    array_6 = []
    array_7 = []
    array_8 = []
    array_9 = []
    array_11 = []
    array_11 = []
    array_12 = []
    in_vae = []
    in_tensor = []
    array_1 = np.array(sequence_input[in_val:out_val])
    array_2 = np.array(reduct_coords_N_y[in_val:out_val])
    array_3 = np.array(reduct_coords_N_z[in_val:out_val])
    array_4 = np.array(reduct_coords_CA_x[in_val:out_val])
    array_5 = np.array(reduct_coords_CA_y[in_val:out_val])
    array_6 = np.array(reduct_coords_CA_z[in_val:out_val])
    array_7 = np.array(reduct_coords_C_x[in_val:out_val])
    array_8 = np.array(reduct_coords_C_y[in_val:out_val])
    array_9 = np.array(reduct_coords_C_z[in_val:out_val])
    array_10 = np.array(reduct_coords_O_x[in_val:out_val])
    array_11 = np.array(reduct_coords_O_y[in_val:out_val])
    array_12 = np.array(reduct_coords_O_z[in_val:out_val])
    in_vae = np.stack((array_1, array_2, array_3, array_4, array_5, array_6, array_7, array_8, array_9, array_10, array_11, array_12), axis = 1)

    in_tensor = torch.from_numpy(in_vae).float()
    return in_tensor

def normalize_array_attom(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val) * 10
    return normalized_array


def plot_atoms(n_array, ca_array, c_array, o_array, sequence, num_epochs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # N atoms
    ax.scatter(n_array[:, 0], n_array[:, 1], n_array[:, 2], c='red', label='N')
    for i in range(len(n_array)):
        ax.text(n_array[i, 0], n_array[i, 1], n_array[i, 2], '', color='red', ha='center', va='center')

    # CA atoms
    ax.scatter(ca_array[:, 0], ca_array[:, 1], ca_array[:, 2], c='green', label='CA')
    for i in range(len(ca_array)):
        ax.text(ca_array[i, 0], ca_array[i, 1], ca_array[i, 2], '', color='green', ha='center', va='center')

    # C atoms
    ax.scatter(c_array[:, 0], c_array[:, 1], c_array[:, 2], c='blue', label='C')
    for i in range(len(c_array)):
        ax.text(c_array[i, 0], c_array[i, 1], c_array[i, 2], '', color='blue', ha='center', va='center')

    # O atoms
    ax.scatter(o_array[:, 0], o_array[:, 1], o_array[:, 2], c='orange', label='O')
    for i in range(len(o_array)):
        ax.text(o_array[i, 0], o_array[i, 1], o_array[i, 2], '', color='orange', ha='center', va='center')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.title("Sequence: " + sequence)  
    plt.show()
    plt.savefig("figure\\sequence_epoch"+str(num_epochs)+".png")

def denormalize_sequence_func(sequence):
    amino_acid_dict = {1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y', 21: 'X', 22: 'B', 23: 'Z', 24: 'J', 25: 'U', 26: 'O'}
    denormalized_sequence = [amino_acid_dict.get(aa, '') for aa in sequence]
    return denormalized_sequence


















