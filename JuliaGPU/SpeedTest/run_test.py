import numpy as numpy
from time import time
import numpy as np
import matplotlib.pyplot as plt

n_calls = 1000


Ns = [10**i for i in range(1, 7)]

from julia import Main
Main.eval('include("PolyKernels.jl")')
#Main.eval("using PolyKernels")

method1 = Main.eval(f"m1(N, Nbasis) = method1(N, Nbasis)")
method2 = Main.eval(f"m2(N, Nbasis) = method2(N, Nbasis)")


Nbasis = 10

times = np.zeros((len(Ns), 2))

# Precompilation
method1(10, 10)
method2(10, 10)

for j, Nbasis in enumerate([10, 100, 1000]):
    for i, N in enumerate(Ns):
        print(N)
        t0 = time()
        [method1(N, Nbasis) for i in range(n_calls)]
        t1 = time()
        times[i, 0] = (t1 - t0) / (n_calls * Nbasis)

        
        t0 = time()
        [method2(N, Nbasis) for i in range(n_calls)]
        t1 = time()
        times[i, 1] = (t1 - t0) / (n_calls * Nbasis)


    plt.plot(Ns, times[:, 0], color=f"C{j+1}", linestyle="solid", label=f"Basis, then multiply approach : {Nbasis=}")
    plt.plot(Ns, times[:, 1], color=f"C{j+1}", linestyle="dashed", label=f"Single Call approach : {Nbasis=}")

plt.xlabel("N_points")
plt.ylabel("Runtime per Basis function (s)")

plt.xscale("log")
plt.yscale("log")

plt.legend()

plt.savefig("Scaling.png")