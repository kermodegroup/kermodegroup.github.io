from   tqdm  import tqdm
import numpy as np

import ase.io
import os

# The XYZ file that was used in the FPS
atoms_filename   = "abcd_all_gen47_20210610-161321.xyz"
atoms_list       = ase.io.read(atoms_filename, index=":")

# The output of the FPS
selected_indexes = np.loadtxt("selected_columns.txt").astype(int)
species          = "C"

# Create a list of index bounds for each structure
index_bounds = []
prev_index   = -1
for atoms in atoms_list:
    length = len([atom for atom in atoms if atom.symbol == species])
    index_bounds.append([prev_index+1, prev_index+length])
    prev_index += length

index_bounds = np.array(index_bounds)

# Logic
# Takes a while...
selected_structures = []
for selected_index in tqdm(selected_indexes):
    for i, (low, high) in enumerate(index_bounds):
        if selected_index >= low and selected_index <= high:
            selected_structures.append(i)

print("{}/{}".format(len(set(selected_structures)),len(selected_structures)))

# unsorted
existing_i = set()
new_atoms_list = []
for i in selected_structures:
    if i not in existing_i:
        new_atoms_list.append(atoms_list[i])
        existing_i.add(i)

ase.io.write("unsorted.FPS-{}.{}".format(len(selected_structures), atoms_filename), new_atoms_list)
print("Done!")
