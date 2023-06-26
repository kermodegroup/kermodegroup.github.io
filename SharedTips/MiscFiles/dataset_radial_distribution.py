import matplotlib.pyplot as plt
from ase.io import read
import numpy as np
from tqdm import trange
from matscipy.angle_distribution import angle_distribution
from matscipy.neighbours import neighbour_list


def get_angular_distributions(ats, cutoff, nbins, radial_bins):
    i, j, d, D = neighbour_list("ijdD", atoms=ats, cutoff=cutoff)

    ang_dist = angle_distribution(i, j, D, nbins)

    radial_counts, bin_centers = np.histogram(d, radial_bins)

    return ang_dist, radial_counts, bin_centers



#### USER PARAMS ####

dataset = read("dataset.xyz", index=":") # Point this at an extXYZ dataset

max_l = 20 # Maximum radial frequency for binning
cutoff = 7.5 # Cutoff Radius for descriptor (r_cut)
nbins = 1000 # Number of bins to use for frequency analysis

#### SYSTEM ####

xticks = np.arange(1, max_l+1, 2)

xticks_minor = np.arange(2, max_l, 2)

yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

ytick_labels = [str(int(tick * 100)) + "%" for tick in yticks]


angles = np.linspace(0, 2*np.pi, nbins)
bins = np.zeros(nbins, dtype=np.int64)

radial_bins = np.linspace(0, cutoff, nbins+1)
rad_bins = np.zeros_like(bins)

for i in trange(len(dataset)):
    res = get_angular_distributions(dataset[i], cutoff, nbins, radial_bins)
    bins += res[0]
    rad_bins += res[1]

plt.plot(2 * np.pi / angles, 1 - np.cumsum(bins)/np.sum(bins))
plt.title(f"Cutoff = {cutoff}")
plt.xlabel("Angular Frequency")
plt.ylabel("CDF")

plt.xticks(xticks)
plt.xticks(xticks_minor, minor=True)
plt.yticks(yticks, labels=ytick_labels)

plt.gca().yaxis.grid(True, which='major')
plt.gca().xaxis.grid(True, which='major')
plt.gca().xaxis.grid(True, which='minor', linestyle="dashed")
plt.xlim(1, max_l)
plt.ylim(0, 1.0)

plt.savefig(f"Angular_Distribution_CDF_rc={cutoff}.png")

plt.clf()

plt.plot(2 * np.pi / angles, (bins)/np.sum(bins))
plt.title(f"Cutoff = {cutoff}")
plt.xlabel("Angular Frequency")
plt.ylabel("PDF")

plt.xticks(xticks)
plt.xticks(xticks_minor, minor=True)

plt.gca().xaxis.grid(True, which='major')
plt.gca().xaxis.grid(True, which='minor', linestyle="dashed")
plt.xlim(1, max_l)

plt.yscale("log")

plt.savefig(f"Angular_Distribution_PDF_rc={cutoff}.png")


plt.clf()



plt.plot(cutoff/res[2][:-1], rad_bins/np.sum(rad_bins))
plt.xlim(0, 6)
plt.yscale("log")
plt.savefig(f"Radial_CDF_rc={cutoff}.png")