# Ising Model at Two Dimension

This repository hots a python implementation of the Ising model in 2 dimensions, exploring phase transition physics.
The simulation allows for:
  + external magnetic field
  + anisotropy between x and y direction
  + more interacting neighboors
  + distance-dependant interaction, based on a user-defined function V(r)

|  Magnetization(T) |  2-points correlation  |
|:-----------------:|:----------------------:|
| ![](plots/PhaseTransition.png) | ![](plots/TwoPointCorr.png)| 


The following shows a 500x500 spin lattice with some perturbations in the four corners.
Below the critical temperature, the domains where spins are aligned tend to grow while
they tend to disappear above the critical temperature.

|  T<Tc |  T>Tc  |
|:-----:|:------:|
| ![](plots/AnimBelowTc.gif) | ![](plots/AnimAboveTc.gif) |


The following shows a 500x500 spin lattice evolving under a temperature T=2.1 for different
number of interacting neighbours. The '1 neighbour' system correpond to the usual Ising model
while 3 and 10 neighbours correponds longer range interactions.

|  1 neighbour | 3 neighbours  |  10 neighbours |
|:-----:|:------:|:-----:|
| ![](plots/AnimInter1.gif) | ![](plots/AnimInter3.gif) | ![](plots/AnimInter10.gif) |

