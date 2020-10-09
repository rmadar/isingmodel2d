import ising

# Create lattice
spins = ising.spinLattice(500)
spins.randomize()

# Add perturbations
spins.perturbate(100, 100, 50)
spins.perturbate(400, 400, 50)
spins.perturbate(100, 400, 50)
spins.perturbate(400, 100, 50)

# Animation below critical temperature
aBelowTc = spins.animate(T=2.1 , nEvolutions=200, saveName='AnimBelowTc')

# Animation above critical temperature
aAboveTc = spins.animate(T=2.5, nEvolutions=200, saveName='AnimAboveTc')
