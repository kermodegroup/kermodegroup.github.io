import sys
import quippy
import ase.io
import numpy as np

from tqdm     import tqdm
from quippy   import descriptors
from operator import itemgetter

#filename = "unsorted.FPS-1743729.abcd_all_gen47_20210610-161321.xyz"
filename = "unsorted.FPS-1743730.abcd_all_gen47_20210610-161321.xyz"

# Len: 181
desc  = descriptors.Descriptor("soap add_species = T l_max = 4 n_max = 8 atom_sigma  = 0.5 cutoff  = 3.5 cutoff_transition_width  = 0.5 central_weight = 1.0")
al    = ase.io.read(filename, index=":") 

# N by SOAP vector length array (not generalised!)
total_SOAPs = np.zeros((N, 181))
prev_index = 0
for a in tqdm(al):
    SOAPs = desc.calc(a)["data"]
    n_atoms = len(a)

    total_SOAPs[prev_index:prev_index+n_atoms,:] = SOAPs

    prev_index += n_atoms


np.save("SOAP_l4-n8-s0.5-c3.5-ctw-0.5-cw-1_{}".format(".".join(filename.split(".")[:-1])), total_SOAPs)
