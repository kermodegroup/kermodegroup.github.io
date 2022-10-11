import sys
import quippy
import ase.io
import numpy as np

from tqdm     import tqdm
from quippy   import descriptors
from operator import itemgetter

filename = "unsorted.FPS-875803.abcd_all_gen47_20210610-161321.xyz"
al       = ase.io.read(filename, index=":")

species             = "C"
total_species_atoms = len([atom for atoms in al for atom in atoms if atom.symbol == species])


# Len: 181
desc  = descriptors.Descriptor("soap add_species = T l_max = 4 n_max = 8 atom_sigma  = 0.5 cutoff  = 3.5 cutoff_transition_width  = 0.5 central_weight = 1.0")
al    = ase.io.read(filename, index=":") 

total_SOAPs = np.zeros((total_species_atoms, 181))
prev_index = 0
for a in tqdm(al):
    SOAPs = desc.calc(a)["data"]
    atom_mask = [i for i, atom in enumerate(a) if atom.symbol == species]
    n_atoms = len(atom_mask)


    new_SOAPs = SOAPs[atom_mask]

    total_SOAPs[prev_index:prev_index+n_atoms,:] = new_SOAPs

    prev_index += n_atoms

    
np.save("{}-ONLY.SOAP_l4-n8-s0.5-c3.5-ctw-0.5-cw-1_{}".format(species, ".".join(filename.split(".")[:-1])), total_SOAPs)
