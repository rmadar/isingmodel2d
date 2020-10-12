import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ising

# Plots style
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'xx-large'
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['figure.figsize'] = (10, 7)

# Spin lattice
spins = ising.spinLattice(100)

# Run phase transition simulator with 100 temperatures points
t, e, m, s = spins.simulatePhaseTransition(nT=100)

# Onsager analytical solution
def onsagerSolution(T):
    return ( 1 - 1./np.sinh(2/T)**4 ) ** (1./8.)

# Plot the result
Tcont = np.linspace(0.1, 2.269, 1000)
plt.plot(t, np.array(m), 'o', label='Simulation')
plt.xlabel('Temperature')
plt.ylabel('Magnetic moment')
plt.plot(Tcont, onsagerSolution(Tcont), label='Onsager\'s solution')
plt.legend()
plt.savefig('../plots/PhaseTransition.png')
