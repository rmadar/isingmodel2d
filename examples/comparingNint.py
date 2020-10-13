import ising

# Animation in the usual Ising model
s1 = ising.spinLattice(500)
a1 = s1.animate(2.1, nEvolutions=150, saveName='../plots/AnimInter1')

# Animation with interaction with 3 closest spins
s2 = ising.spinLattice(500, Nint=3)
a2 = s2.animate(2.1, nEvolutions=150, saveName='../plots/AnimInter3')

# Animation with interaction with 10 closest spins
s3 = ising.spinLattice(500, Nint=10)
a3 = s3.animate(2.1, nEvolutions=150, saveName='../plots/AnimInter10')
