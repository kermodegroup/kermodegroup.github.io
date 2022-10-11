# -*- coding: utf-8 -*-
import os
import rdf
import sys
import ase.io
import argparse
import pandas            as pd
import seaborn           as sns
import numpy             as np
import networkx          as nx
import matplotlib.pyplot as plt
from ase.neighborlist    import NeighborList
from matplotlib.ticker   import MaxNLocator
from itertools           import cycle
from tqdm                import tqdm

#sys.setrecursionlimit(10**6)

#==============================================================================
# Funtions
#==============================================================================
def add_neighbour(G, node, lpath, prepend=False):
    check_lpath = lpath.copy()
    if node in check_lpath:
        check_lpath.remove(node)
        
    for n in G.neighbors(node):
        if n not in lpath:
            can_add = True
            # If we have 2 or mor neighbours see if we've missed a loop atom
            #
            # With one neighbour, we can't have missed a neighbour, and none
            #  means the atom is isolated
            if G.degree[n] >= 2:
                for nn in G.neighbors(n):
                    if nn in check_lpath:
                        can_add = False
                        
                if can_add:
                    if prepend:
                        lpath.insert(0, n)
                    else:
                        lpath.append(n)
                        
                    # Can we add another neighbour
                    lpath = add_neighbour(G, n, lpath, prepend)
        
    return lpath

#------------------------------------------------------------------------------
def check_chain(G, T, lpath):
    if len(lpath) != 0:
        new_T = nx.dfs_tree(G, lpath[-1])
        new_lpath = nx.dag_longest_path(new_T)
        if len(new_lpath) > len(lpath):
            lpath = check_chain(G, T, new_lpath)
        else:
            # Neighbour checking at the ends of the path
            lpath = add_neighbour(G, lpath[0],  lpath, True)
            lpath = add_neighbour(G, lpath[-1], lpath, False)


            
    return lpath

#------------------------------------------------------------------------------
def check_loop(G, lpath):
    if len(lpath) != 0:
        for i, node in enumerate(lpath):
            # There can be no loop with only one or two atoms
            if i != 0 and i != 1:
                if G.degree[node] >= 2:
                    for n in G.neighbors(node):
                        if n in lpath[:i-1]: # i-1 becuase we don't want to find the atom one back in the chain
                            return lpath[:i]
            
    return lpath
            
#------------------------------------------------------------------------------
def find_chains(chains, G, atoms, total_chains, nodes):
    skipped_nodes = []
    for source in nodes:
        verbose = False
        if source == 2464 or source == 2465 or source == 2471:
            verbose = True

        if source in G:
            # Do the DFS
            T = nx.dfs_tree(G, source)
            
            degrees = list(dict(T.degree).values())
            if 1 in degrees:
                #--------------------------------------------------------------
                # Find the longest path
                #--------------------------------------------------------------
                lpath = nx.dag_longest_path(T)

                #----------------------------------------------------------
                # Check if it is the longest path in the whole sub-graph
                #----------------------------------------------------------
                lpath = check_chain(G, T, lpath)
    
    
                #----------------------------------------------------------
                # Check if it there are any loops and shorten the chain
                #----------------------------------------------------------
                lpath = check_loop(G, lpath)  
    

  
                #----------------------------------------------------------
                # Save some stuff to the extxtz
                #----------------------------------------------------------
                atoms.arrays["chain_id"][lpath]     = total_chains
                atoms.arrays["chain_length"][lpath] = len(lpath)
                atoms.arrays["chain_pos"][lpath]    = np.arange(0,len(lpath), 1)
                total_chains += 1        
 
                # Update the dictionary
                if len(lpath) in chains.keys():
                    chains[len(lpath)] += 1
                else:
                    chains[len(lpath)] = 1
                        
                
                #----------------------------------------------------------
                # Remove selected nodes from the graph, we don't want to
                #  use them in another chain
                #----------------------------------------------------------
                G.remove_nodes_from(lpath)
                    
                        
                        
            elif T.degree[source] == 0:
                # chain of lenge 0 (isolated atom)
                # Update the dictionary
                if 1 in chains.keys():
                    chains[1] += 1
                else:
                    chains[1] = 1
                    
                G.remove_nodes_from([source])
                
                
            # We have no degree 1 nodes
            else:
                # Get the longest path from wherever we are
                tmp_lpath = nx.dag_longest_path(T)
                
                # and your the end node to make a new rooted tree from the end 
                #  of the interim longest path
                #
                # Making the tree from the undirected tree stops the first node
                T2 = nx.dfs_tree(T.to_undirected(), tmp_lpath[-1])
                
                # This is the longest path from the remaining 
                lpath = nx.dag_longest_path(T2)
                
                #----------------------------------------------------------
                # Check if it there are any loops and shorten the chain
                #----------------------------------------------------------
                lpath = check_loop(G, lpath)
                
                
                ##
                #----------------------------------------------------------
                # Save some stuff to the extxtz
                #----------------------------------------------------------
                atoms.arrays["chain_id"][lpath]     = total_chains
                atoms.arrays["chain_length"][lpath] = len(lpath)
                atoms.arrays["chain_pos"][lpath]    = np.arange(0,len(lpath), 1)
                total_chains += 1        
 
                # Update the dictionary
                if len(lpath) in chains.keys():
                    chains[len(lpath)] += 1
                else:
                    chains[len(lpath)] = 1
                        
                
                #----------------------------------------------------------
                # Remove selected nodes from the graph, we don't want to
                #  use them in another chain
                #----------------------------------------------------------
                G.remove_nodes_from(lpath)
                
    
    # outside loop    
    if len(skipped_nodes) != 0:
        chains = find_chains(chains, G, atoms, total_chains, skipped_nodes)

    return chains
#==============================================================================  
    
#-----------------------#
#   Bonding for SiC     #
# C - C : 1.6  Angstrom #
# Si- C : 1.89 Angstrom #
# Si-Si : 2.35 Angstrom #
#-----------------------#

parser = argparse.ArgumentParser()
parser.add_argument("filename",            metavar="f",                                  type=str,   help="The NetCDF trajectory file")
parser.add_argument("--stride",            metavar="-s", nargs="?", default=100,         type=int,   help="The stride to take the trajectory at")

parser.add_argument("--max_chain",         metavar="-m", nargs="?", default=45,          type=int,   help="The maximum length of chain that will be stored")
parser.add_argument("--max_chain_display", metavar="-M", nargs="?", default=15,          type=int,   help="The maximum length of chain that will be shown on the graph")
parser.add_argument("--elements",          metavar="-e", nargs="+", default=["C","Si"],  type=str,   help="The elements we wish to perform chain analysis on")
parser.add_argument("--element_cutoffs",   metavar="-c", nargs="+", default=[1.6, 2.35], type=float, help="The element cutoff values for the provided elements")


args            = parser.parse_args()
max_chain       = args.max_chain         
graph_max_chain = args.max_chain_display 
elements        = args.elements          
element_cutoffs = args.element_cutoffs   

if len(elements) != len(element_cutoffs):
    sys.exit("The number of cutoffs must match the number of elements provided") 

structure       = [[element, cutoff, [not_element for not_element in elements if not_element != element]] for element, cutoff in zip(elements, element_cutoffs)]

all_chains_dict = {}
al_dict         = {}

for element, cutoff, bad_species in structure:
    if os.path.isfile("{}_chains_{}.txt".format(element.lower(), ".".join(args.filename.split(".")[:-1]))):
        all_chains_dict[element] = np.loadtxt("{}_chains_{}.txt".format(element.lower(), ".".join(args.filename.split(".")[:-1])))
        al_dict[element]         = ase.io.read("chain-marked_{}.{}.xyz".format(element, ".".join(args.filename.split(".")[:-1])), index=":")

    else:
        # Load the Atoms list from either netCDF or extxyz
        #  there is no reason why it can't be loaded from any ase reader, just not implemented properly
        if args.filename.split(".")[-1] == "nc":
            al_dict[element] = rdf.lammps2atoms(ase.io.NetCDFTrajectory(args.filename, "r")[::args.stride])
        elif args.filename.split(".")[-1] == "xyz":
            al_dict[element] = ase.io.read(args.filename, index="::{}".format(args.stride))
        else:
            sys.exit("Incorrect filename extension, neither `.nc` or `.xyz`")

        # Build empty chains matrix
        all_chains_dict[element]  = np.zeros((len(al_dict[element]),max_chain), dtype=int)

        # Loop!
        for frame, atoms in enumerate(tqdm(al_dict[element], desc="Working on element - {:>2s}".format(element))):
            del atoms[[atom.index for atom in atoms if atom.symbol in bad_species]]

            total_chains                 = 1
            atoms.arrays["chain_id"]     = np.zeros(len(atoms))
            atoms.arrays["chain_length"] = np.zeros(len(atoms))
            atoms.arrays["chain_pos"]    = np.zeros(len(atoms))
            cutoffs                      = [cutoff/2]*len(atoms)
            
            # Graph building
            nl = NeighborList(cutoffs, skin=0, self_interaction=False, bothways=False)
            nl.update(atoms)
            cm = nl.get_connectivity_matrix()
            G = nx.convert_matrix.from_scipy_sparse_matrix(cm)
            
            # Find out chains, and store them in a handy dictionary
            chains = find_chains({}, G, atoms, total_chains, range(len(atoms)))
            

            for key in sorted(chains.keys()):
                if key >= all_chains_dict[element].shape[1]:
                    print("Warning: There is a chain ({}) larger than the storage matrix ({})".format(key, all_chains_dict[element].shape[1]))
                else:
                    all_chains_dict[element][frame,key] = chains[key]
            
        ase.io.write("chain-marked_{}.{}.xyz".format(element, ".".join(args.filename.split(".")[:-1])), al_dict[element])
        np.savetxt("{}_chains_{}.txt".format(element.lower(), ".".join(args.filename.split(".")[:-1])), all_chains_dict[element])

total_al = []
# Zip up all the parts of the original atoms object
for elementwise_al in zip(*al_dict.values()):

    # Add the first element into the atoms
    atoms = elementwise_al[0]

    # For all additional elements add them in
    for ind_atoms in elementwise_al[1:]:
        atoms += ind_atoms
    
    # Save out the recombined elements
    total_al.append(atoms.copy())

ase.io.write("chain-marked_all.{}.xyz".format(".".join(args.filename.split(".")[:-1])), total_al)

#==============================================================================
# Plotting
#==============================================================================
fontsize    = 16

colours     = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
styles      = ["-", "-.", "--", ":"]
linecyclera = cycle(["k"] + [c+s for s in styles for c in colours])
linecyclerb = cycle([c+s for s in styles for c in colours])


for element in ["C", "Si"]:
    print("Working on element: {}".format(element))
    fig, axes = plt.subplots(1,2,figsize=(20,10))
    time = None
    if "time" in al_dict[element][0].info.keys() and "time" in al_dict[element][-1].info.keys():
        time      = np.linspace(float(al_dict[element][0].info["time"]), float(al_dict[element][-1].info["time"]), len(al_dict[element]))
    else:
        time = np.arange(0,len(al_dict[element]))
    not_plot = []
    for i in range(1, max_chain):
        if all_chains_dict[element][:,i].sum() != 0:
            axes[0].plot(time, all_chains_dict[element][:,i], next(linecyclera), label="{}".format(i))
            if i != 1:
                axes[1].plot(time, all_chains_dict[element][:,i], next(linecyclerb), label="{}".format(i))
        else:
            not_plot.append(i)

    print(" No chains length {}, not plotting".format(not_plot))
    
    
    axes[0].set_title("Chains (Including Isolated Atoms)", fontsize=fontsize+2)
    axes[1].set_title("Chains", fontsize=fontsize+2)
    
    for ax in axes.flatten():
        ax.legend(fontsize=fontsize, fancybox=True, shadow=True, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), title="Chain Length") #, title_fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Length of Chain", fontsize=fontsize)
        ax.set_xlabel("Time [ps]", fontsize=fontsize)
        ax.grid()
        
        ax.set_xlim((time[0], time[-1]))
    
    print()
    fig.savefig("chains_{}_{}.png".format(element, ".".join(args.filename.split(".")[:-1])), dpi=300, bbox_inches="tight")

#------------------------------------------------------------------------------
# Chains Hist
#------------------------------------------------------------------------------
fig_bar, ax_bar = plt.subplots(1,1, figsize=(10,10))

df_data = []
for element in elements:
    data = [(index, value, element) for index, _list in enumerate(all_chains_dict[element].T) if index < graph_max_chain for value in _list]
    for item in data:
        df_data.append(item)

df     = pd.DataFrame(df_data, columns=["Chain Length", "Count", "Element"])
ax_bar = sns.barplot(x="Chain Length", y="Count", hue="Element", data=df, ax=ax_bar, estimator=np.mean, ci="sd", capsize=0.2)


ax_bar.set_title("Average Number of Chains per Chain Length", fontsize=fontsize+2)

ax_bar.set_xlabel("Chain Length", fontsize=fontsize)
ax_bar.set_ylabel("Average Number of Chains", fontsize=fontsize)

ax_bar.tick_params(axis="both", labelsize=fontsize)
ax_bar.set_yscale("log")
ax_bar.set_axisbelow(True)
ax_bar.grid()

ax_bar.set_ylim(bottom=0)

fig_bar.tight_layout(pad=3)

fig_bar.savefig("chains_bar_{}.png".format(".".join(args.filename.split(".")[:-1])), dpi=300, bbox_inches="tight")
