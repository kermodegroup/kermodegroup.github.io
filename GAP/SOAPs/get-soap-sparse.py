import sys
import numpy as np
import ase.io


#-----------------------------------------------------------------------
# User Variables 
#-----------------------------------------------------------------------
atoms_filename  = "../in.training.xyz"        # The filename of the ase atoms_list used to train
name            = "G52_sparse-per-config"     # The name you wish to save out the datafile with
sparse_filename = "soap-sparse-indexes.dat"   # The name of the gap_fit soap sparse indexes file
key1            = "config_type"               # The first key you want data collected for from the atoms info dictionary
key2            = "generation"                # The second key you want data collected for from the atoms info dictionary



#-----------------------------------------------------------------------
al              = ase.io.read(atoms_filename, index=":")

#=======================================================================
# Create a dictonary of total number of structures in each layer
#=======================================================================
def total_atoms_in_env(value1, value2, total_atoms, natoms):
    """
    Create a double layer dictionary with the total number of atoms for each entry
    
    Variables:
     value1      - str/int - The value contained in the atoms info object of key1
     value2      - str/int - The value contained in the atoms info object of key2
     total_atoms - dict    - The double layer dictionary we are building
     natoms      - int     - The number of atoms in the current atoms object (for our species)
    """
    if value1 in total_atoms.keys():
        if type(total_atoms[value1]) is dict:
            if value2 in total_atoms[value1].keys():
                total_atoms[value1][value2] += natoms
            else:
                total_atoms[value1][value2] = natoms
        else:
            print("No second nested dictionary! Aborting (key1: {} | key2: {})".format(key1, key2))
            sys.exit()
    else:
        total_atoms[value1] = {value2 : no_c}
#-----------------------------------------------------------------------

total_confs    = {}
total_atoms_c  = {}
total_atoms_si = {}

# Dual layer dictionary, containing the total number of structures/atoms
for a in al:
    value1 = a.info[key1]
    value2 = a.info[key2]
    
    no_c  = len([x for x in a.numbers if x == 6])
    no_si = len([x for x in a.numbers if x == 14])

    #-------------------
    # Structures
    #-------------------
    if value1 in total_confs.keys():
        if type(total_confs[value1]) is dict:
            if value2 in total_confs[value1].keys():
                total_confs[value1][value2] += 1
            else:
                total_confs[value1][value2] = 1
        else:
            print("No second nested dictionary! Aborting (key1: {} | key2: {})".format(key1, key2))
            sys.exit()
    else:
        total_confs[value1] = {value2 : 1}

    #-------------------
    # C Envs
    #-------------------
    total_atoms_in_env(value1, value2, total_atoms_c, no_c)
    
    #-------------------
    # Si Envs
    #-------------------
    total_atoms_in_env(value1, value2, total_atoms_si, no_si)

#-----------------------------------------------------------------------

#=======================================================================
# Create a dictonary of total number of sparse points for each category
#=======================================================================
def find_sparse_points(a, atoms_index, index, nsparse, sparse_indexes, sp_index, data, key1, key2, config_dict, count, natoms):
    """
    Create a double layer dictionary with the number of sparse points in each category

    Variables:
     a              - Atoms Object 
     atoms_index    - The index of the atoms list where the atoms object comes from
     index          - the index of the sparse list we start from
     nsparse        - the number of sparse points total
     sparse_indexes - the list of indexes from the atoms list where a sparse point is located
     sp_index       - the index for the sparse point in that atoms object
     data           - the list if sparse indexes for the species we are working on
     key1           - the key for the first layer of the nested dictionary
     key2           - the key for the second layer of the nested dictionary
     config_dict    - the double layer nested dictionary
     count          - the total number of atoms we have seen so far
     natoms         - the number of atoms in the current atoms object of the species we are looking at
    """
    # We need to catch multiple instances in one atoms object
    for i in range(index, nsparse, 1):
        # If the next carbon index is in this atoms object save some stuff
        if data[i] < count + natoms:
            # Add to the dictionary
            value1 = a.info[key1]
            value2 = a.info[key2]
            
#             #=====================================
#             if value1 == "dimer":
#                 print(" C", i, a.info["bond_length"], a[0].symbol, a[1].symbol)
#             #=====================================
            
            sp_index.append(data[i] - count)
            sparse_indexes.append(atoms_index)
            if value1 in config_dict.keys():
                if type(config_dict[value1]) is dict:
                    if value2 in config_dict[value1].keys():
                        config_dict[value1][value2] += 1
                    else:
                        config_dict[value1][value2] = 1
                else:
                    print("No second nested dictionary! Aborting (key1: {} | key2: {})".format(key1, key2))
                    sys.exit()
            else:
                config_dict[value1] = {value2 : 1}
            
            # Increment the index to look for the next value
            index += 1
        else:
            break
            
    return index
#-----------------------------------------------------------------------

data              = np.loadtxt(sparse_filename, dtype=int)
sparse_points     = data.shape[-1] # nsparse

c_index           = 0
c_data            = []
c_count           = 0
c_config_dict     = {}
sparse_indexes_c  = []
sp_indexes_c      = []

si_index          = 0
si_data           = []
si_count          = 0
si_config_dict    = {}
sparse_indexes_si = []
sp_indexes_si     = []

for j, a in enumerate(al):    
    no_c  = len([x for x in a.numbers if x == 6])
    no_si = len([x for x in a.numbers if x == 14])
    
    #-------------------
    # C Envs
    #-------------------
    c_index = find_sparse_points(a, j, c_index, sparse_points, sparse_indexes_c, sp_indexes_c, data[1,:], key1, key2, c_config_dict, c_count, no_c)
    
    #-------------------
    # Si Envs
    #-------------------
    si_index = find_sparse_points(a, j, si_index, sparse_points, sparse_indexes_si, sp_indexes_si, data[0,:], key1, key2, si_config_dict, si_count, no_si)
        
    c_count  += no_c
    si_count += no_si


# Create an atoms list for the structures that contain sparse points
c_al  = [al[i] for i in sorted(list(set(sparse_indexes_c)))]
si_al = [al[i] for i in sorted(list(set(sparse_indexes_si)))]

np.savetxt("c_struc_indexes.txt",  np.array(list(set(sparse_indexes_c))).astype(int))
np.savetxt("si_struc_indexes.txt", np.array(list(set(sparse_indexes_si))).astype(int))

# Save said atoms lists
ase.io.write("c_sparse.xyz", c_al)
ase.io.write("si_sparse.xyz", si_al)


c_al_sparse = []
print(sp_indexes_c)
for a in range(len(sparse_indexes_c)):
    moda = al[sparse_indexes_c[a]].copy()
    siis = [i for i, a in enumerate(moda) if a.symbol == "Si"]

    del moda[siis] 

    print(siis)
    print("index: {:>3d} | atom index: {:>3d} | frame index: {:>5d} | atom: {}".format(a, sp_indexes_c[a], sparse_indexes_c[a], moda[sp_indexes_c[a]]))

    if "sp" not in moda.arrays.keys():
        moda.arrays["sp"] = np.zeros(len(moda))

    moda.arrays["sp"][sp_indexes_c[a]] = 1
    c_al_sparse.append(moda)

ase.io.write("tmp.xyz", c_al_sparse)
sys.exit()

#=======================================================================
# Print the Data! (This is not as general!)
#=======================================================================
cc_format_string = "{:>20s}: {:>20s}"
cc_title = cc_format_string.format(key1, key2)

print("{:>42s} | {:^11s} | {:^23s} | {:>27s} | {:>27s} |".format("", "All", "All Environments", "", ""))
print("{:>42s} | {:^11s} | {:^10s} | {:^10s} | {:^27s} | {:^27s} |".format(cc_title, "Configs", "C", "Si", "C Selected (All/Env.)", "Si Selected (All/Env.)", ))
print("-"*143+"|")
csvs = []
# Header
csvs.append("Helper,Config Type,Generation,Number of  Configurations,Number of C Environments,Number of Si Environments,Selected C Environments,C Percentage out of Configurations,C Percentage out of Environments,Selected Si Environments,Si Percentage out of Configurations,Si Percentage out of Environments\n")
            
for value1 in sorted(total_confs.keys()):
    for value2 in sorted(total_confs[value1].keys()):
        if value1 != value2:
            compound_key = cc_format_string.format(str(value1), str(value2))
        else:
            compound_key = str(value1)

        c_count  = 0
        si_count = 0
        if value1 in c_config_dict.keys():
            if value2 in c_config_dict[value1].keys():
                c_count = c_config_dict[value1][value2]


        if value1 in si_config_dict.keys():
            if value2 in si_config_dict[value1].keys():
                si_count = si_config_dict[value1][value2]

        # Calculate percentages
        c_all_percent = (c_count/total_confs[value1][value2])*100
        if total_atoms_c[value1][value2] != 0:
            c_env_percent = (c_count/total_atoms_c[value1][value2])*100
        else:
            c_env_percent = 0

        si_all_percent = (si_count/total_confs[value1][value2])*100
        if total_atoms_si[value1][value2] != 0:
            si_env_percent = (si_count/total_atoms_si[value1][value2])*100
        else:
            si_env_percent = 0
        csvs.append("{},{},{},{},{},{},{},{},{},{},{},{}\n".format("{}{}".format(value1,value2),
                                                                   value1,
                                                                   value2,
                                                                   total_confs[value1][value2],
                                                                   total_atoms_c[value1][value2],
                                                                   total_atoms_si[value1][value2],
                                                                   c_count,
                                                                   c_all_percent,
                                                                   c_env_percent,
                                                                   si_count,
                                                                   si_all_percent,
                                                                   si_env_percent))
        
        print("{:>42s} | {:>11d} | {:>10d} | {:>10d} | {:>5d} ({:>6.2f} %) ({:>6.3f} %) | {:>5d} ({:>6.2f} %) ({:>6.3f} %) |".format(compound_key,
                                                                                                                                   total_confs[value1][value2],
                                                                                                                                   total_atoms_c[value1][value2],
                                                                                                                                   total_atoms_si[value1][value2],
                                                                                                                                   c_count,
                                                                                                                                   c_all_percent,
                                                                                                                                   c_env_percent,
                                                                                                                                   si_count,
                                                                                                                                   si_all_percent,
                                                                                                                                   si_env_percent))

#=======================================================================
# Save the data
#=======================================================================
with open("{}.csv".format(name), "w") as f:
    f.writelines(csvs)
