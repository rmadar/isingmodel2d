import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats, optimize
import pandas as pd
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

# Create a spin lattice
s = ising.spinLattice(30)

# Compute correlation function for different temperatures
Ts = [0.5, 1., 1.5, 1.75, 2, 2.1, 2.15, 2.175, 2.2, 2.225, 2.25, 2.269, 2.30, 2.35, 2.40, 2.5, 2.6, 2.8, 3, 3.5, 4]
Rs, Gs = [], []
for T in Ts: 
    
    # Start from aligned spins
    s.align()
    
    # Compute 2 points correlation function
    r, g = s.twoPtsCorr(T, nEvolution=20000)

    # Store the results
    Rs.append(r)
    Gs.append(g)


# Loop over temperatures, compute correpsonding correlations
# averaged over radius, and save the result in a dataframe
df = pd.DataFrame()
for r, g, T in zip(Rs, Gs, Ts):
    
    # Bin data
    gAve, binsEdges, binNumber = stats.binned_statistic(r, g, statistic='mean', bins=25)
    gRms, binsEdges, binNumber = stats.binned_statistic(r, g, statistic=lambda a: a.std(), bins=25)
    gSum, binsEdges, binNumber = stats.binned_statistic(r, g, statistic='sum', bins=25)
    
    # Compute 
    rAve = (binsEdges[1:] + binsEdges[:-1])/2.0
    df['r'] = rAve
    df['MeanT' + str(T)] = gAve
    df['RmsT' + str(T)]  = gRms
    df['SumT' + str(T)]  = gSum

    
# Plot the correlation function for different temperatures
for T in [1., 2, 2.25, 2.50, 4]:
    x, y, dy = df['r'], df['MeanT' + str(T)], df['RmsT' + str(T)]
    plt.plot(x, y)
    plt.fill_between(x, y1=y-dy, y2=y+dy, alpha=0.3, label='T='+str(T))
plt.xlabel('$r_{ij}$')
plt.ylabel('$C(r_{ij}) \\equiv <\\sigma_{i}\\sigma_{j}> - <\\sigma_{i}>^{2}$')
plt.legend()
plt.savefig('../plots/TwoPointCorr.png')


# Perform an exponential fit and plot the correlation
# length as function of the temperature
def expDecay(x, A, D, X0):
    return A * np.exp( -(x-X0)/D )

# Loop over temperaturews
corrDist = []
for T in Ts:
    x, y = df['r'], df['MeanT' + str(T)]
    p, cov = optimize.curve_fit(expDecay, x, y, p0=[0.25, 0.2, 1], bounds=([0.0, 0.0, 0.5], [1, 20, 2]))
    corrDist.append(p[1])

plt.figure()
plt.plot(Ts, corrDist, markersize=10, marker='o', linewidth=3)
plt.ylabel('Correlation distance')
plt.xlabel('Temperature')
plt.savefig('../plots/corrLength.png')
