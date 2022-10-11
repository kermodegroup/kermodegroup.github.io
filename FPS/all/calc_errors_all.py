import os
import ase.io

import numpy as np

from scipy.linalg import pinv2
from tqdm         import tqdm

# The XYZ file from the FPS (ordered)
u_al = ase.io.read("unsorted.FPS-1743729.abcd_all_gen47_20210610-161321.xyz", index=":")
indexes = [0]
for a in u_al:
    indexes.append(indexes[-1] + len(a))
    
del indexes[0]

# The indexes for each structure as a cumsum
np.savetxt("col_indexes.txt", np.array(indexes, dtype=int))

# The entire descriptor matrix
desc_data = np.load("SOAP_l4-n8-s0.5-c3.5-ctw-0.5-cw-1_unsorted.FPS-1743729.abcd_all_gen47_20210610-161321.npy")
data      = desc_data.T
print(data.shape)

norm_data = np.linalg.norm(data)

# Pesudo inverse of the data
print("Inversing the matrix")
pinv_data = pinv2(data)

# Matrix multiplication of data*pinv_data
print("Multiply Matracies")
XXp       = np.dot(data, pinv_data)


if os.path.isfile("struc_errors.txt"):
    errors = np.loadtxt("struc_errors.txt")
else:
    errors = np.zeros(len(indexes))

for j, i in enumerate(tqdm(indexes)):
    if errors[j] != 0:
        continue

    # Selected column(s)
    if i == 0:
        _C = data[:, 0].reshape(-1,1)
    else:
        _C = data[:, 0:i]
        
    _R = data 
    _U = np.dot(pinv2(_C), XXp)

    errors[j] = np.linalg.norm(data - np.dot(np.dot(_C, _U), _R)) / norm_data
    
    # Checkpointing
    if j % 10 == 0:
        np.savetxt("struc_errors.txt", errors)
np.savetxt("struc_errors.txt", errors)
