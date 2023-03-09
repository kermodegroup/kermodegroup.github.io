import numpy as np
from julia.api import Julia
from ase.atoms import Atoms
jl = Julia(compiled_modules=False)

from julia import Main
Main.eval('include("ace_descriptor.jl")')
Main.eval("using ASE, JuLIP, ACE1pack, .ace_descriptor")
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")
ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")

jl_ace = Main.eval("get_ace(species, N, maxdeg, rcut) = ace_descriptor.ACE(species, N, maxdeg, rcut)")
jl_get_vecs = Main.eval("get_vecs(ace, ats) = ace_descriptor.get_ace_vecs(ace, ats)")
jl_get_basis_len = Main.eval("get_len(ace) = length(ace)")

def _ACE(species, N=3, max_deg=3, rcut=5.0):
    ace = jl_ace(species, N, max_deg, rcut)
    return ace

def get_ace_vecs(ace, atoms):
    jl_ats = ASEAtoms(atoms)
    jl_ats = convert(jl_ats)

    vecs = jl_get_vecs(ace, jl_ats)

    return vecs



class ACE():
    def __init__(self, species, N=3, max_deg=3, rcut=5.0):
        '''
        Build a Julia ACE model with a Python interface
        Intended only for computing ACE Descriptor vectors on a structure

        ### Args ###
        species : List of strings; Chemical symbols for the ACE model
                    EG: species=["In", "P"]

        N       : int; Correlation order

        max_deg : int; Maximum polynomial degree

        rcut    : float; Radial cutoff in Ang for the basis
        '''
        self.species = species
        self.N = N
        self.max_deg = max_deg
        self.rcut = rcut
        self.desc = _ACE(species, N, max_deg, rcut)
        self.basis_len = jl_get_basis_len(self.desc)

    def get_ace_vecs_single(self, struct):
        return get_ace_vecs(self.desc, struct)

    def get_ace_vecs_dataset(self, images):
        return [self.get_ace_vecs_single(image) for image in images]

    def __call__(self, atoms):
        if type(atoms) == Atoms:
            # Single image
            return self.get_ace_vecs_single(atoms)
        elif type(atoms) == list:
            # List of images
            return self.get_ace_vecs_dataset(atoms)

    def __len__(self):
        return self.basis_len



if __name__ == "__main__":
    from ase.io import read
    bulk = read("Bulk/ZB_Bulk.xyz", index="-1")

    ace = ACE(["In", "P"], N=3, max_deg=3, rcut=5.0)

    print(ace(bulk).shape)
