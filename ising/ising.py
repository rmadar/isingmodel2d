import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class spinLattice:
    
    def __init__(self, N, externalField=0, Jx=1, Jy=1, Nint=0):
        '''
        Create an object 2D spinLattice of size NxN with an 
        external field H (0 by default), and two coupling Jx, 
        Jy in both directions (1 by default). The energy of 
        the system is modified by the external field by adding
        a term -extH*sum(s).

        Arguments:
        ----------
        N: lattice of size NxN
        externalField: intensity of the external magnetic field
        Jx: coupling constant in the x-axis
        Jy: coupling constant in the y-axis
        Nint: number of interacting neighboor spins.
        '''
        
        self.spins  = 2*np.random.randint(2, size=(N,N))-1
        self.extH   = externalField
        self.Jx     = Jx
        self.Jy     = Jx
        self.Nint   = Nint
        self.cost   = self.costIsing
        self.energy = self.energyIsing
        if self.Nint > 0:
            self.cost = self.costNint
            self.energy = self.energyNint

            
    def align(self):
        '''
        Align all spins together
        '''
        self.spins = np.zeros_like(self.spins) + 1

        
    def antialign(self):
        '''
        Anti-align all spins. Possible if N is even (raise an error otherwise).
        '''
        N = self.spins.shape[0]
        if N%2 != 0:
            raise NameError('Anti-alligned spin system must have an even N')
        l1 = np.zeros(N) + 1
        l1[::2] = -1
        l2 = np.zeros(N) + 1
        l2[1::2] = -1
        self.spins = np.array([l1, l2]*int(N/2))
        

    def randomize(self):
        '''
        Randomely flip spins.
        '''
        flip = 2*np.random.randint(2, size=self.spins.shape)-1
        self.spins *= flip

        
    def perturbate(self, x0, y0, R, s=-1):
        '''
        Put all spins in s state (-1 by default) in a give 
        region defined as a circle centered in (x0, y0) and
        of radius R. x0, y0 and R are expressed in lattice
        unit.
        '''
        N = self.spins.shape[0]
        x, y = np.meshgrid(np.arange(N), np.arange(N))
        x, y = x.flatten(), y.flatten()

        d = np.sqrt((x-x0)**2  + (y-y0)**2)
        xPert, yPert = x[d<R], y[d<R]
        self.spins[xPert, yPert] = s

        
    def plot(self):
        '''
        Plotting a snapshot of the 2D spins lattice.
        '''
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        plt.imshow(self.spins.copy(), cmap='BuPu', origin='bottom', vmin=-1.0, vmax=1.0)


    def energyIsing(self):
        '''
        Compute and return the energy of the lattice, 
        normalized to the number of spins.
        '''
    
        # Prepare shifted arrays for vectorized computation
        sPad = np.pad(self.spins,  (1, 1)) # Padding with 0
        sdY  = sPad[0:-2, 1:-1]            # Bring neighbour in y to the current node
        sdX  = sPad[1:-1, 0:-2]            # Bring neighbour in x to the current node
        
        # energy = -1/4 s*up + s*down + s*left + s*right
        energy = -1./2. * self.spins * ( self.Jx*sdX + self.Jy*sdY )
        
        # Sum over the lattice and energy due to external field
        return energy.sum() / self.spins.size - self.extH * self.magnetization()


    def energyNint(self):
        '''
        Compute the energy in case on N interactions.
        '''
        
        # Container for the final energy computation
        N = self.Nint
        energy = np.zeros_like(self.spins, dtype=np.float64)
    
        # Prepare shifted arrays for vectorized computation                              
        sPad = np.pad(self.spins, (N, N)) # Padding with 0                              
        
        Npairs = 0
        for i in range(N+1):
            for j in range(N+1):
                if (i, j) == (0, 0): continue
                if abs(i) == abs(j): continue
                J = (i*self.Jx + j*self.Jy) / np.sqrt(i**2+j**2)
                yStart, yEnd = N+j, -N+j
                xStart, xEnd = N+i, -N+i
                if xEnd == 0:
                    xEnd = None
                if yEnd == 0:
                    yEnd = None
                    
                energy += - J * self.spins * sPad[yStart:yEnd, xStart:xEnd]
                Npairs += 1
        
        # Remove double counting of since (i, j) and (i, j) should be counted once.
        energy /= Npairs                                                                               
        
        # Sum over the lattice and energy due to external field
        return energy.sum() / self.spins.size - self.extH * self.magnetization()
    
    
    def magnetization(self):
        '''
        Compute and return the magnetization of the spin lattice,
        normalized to the number of spins.
        '''
        return self.spins.sum() / self.spins.size


    def entropy(self):
        '''
        Compute and return the entropy of the system. For a spin-1/2 system, the
        number of configuration for a given microstate is N! / (N[up]! N[down]!).
        Due to numerical consideration, the Stirling’s approximation is used:
           ln(x!) ~ x ln(x) - x
        '''
        Nup = np.count_nonzero(self.spins== 1)
        Ndo = np.count_nonzero(self.spins==-1)    
        if Nup != 0 and Ndo !=0 :
            return (Nup+Ndo) * np.log(Nup+Ndo) - Nup * np.log(Nup) - Ndo * np.log(Ndo)
        else:
            return 0


    def costNint(self):
        '''
        Return the cost in case of N interactions.
        '''
        
        # Container for the final energy computation
        N = self.Nint
        cost = np.zeros_like(self.spins, dtype=np.float64)
    
        # Prepare shifted arrays for vectorized computation                              
        sPad = np.pad(self.spins, (N, N)) # Padding with 0
        
        for i in range(-N, N+1):
            for j in range(-N, N+1):
                if (i, j) == (0, 0): continue
                if abs(i) == abs(j): continue
                J = (abs(i)*self.Jx + abs(j)*self.Jy) / np.sqrt(i**2+j**2)
                yStart, yEnd = N+j, -N+j
                xStart, xEnd = N+i, -N+i
                if xEnd == 0:
                    xEnd = None
                if yEnd == 0:
                    yEnd = None
                    
                cost += 2 * J * self.spins * sPad[yStart:yEnd, xStart:xEnd]
                                             
        return cost - 2 * self.spins * self.extH 

    
    def costIsing(self):
        '''
        Return the cost in case of vanilla Ising model.
        '''

        # Prepare shifted arrays for vectorized computation
        sPad = np.pad(self.spins, (1, 1))     # Padding with 0
        sUp = sPad[0:-2, 1:-1]                # Bring up neighbour to the current node
        sDo = sPad[2:  , 1:-1]                # Bring down neighbour to the current node
        sLe = sPad[1:-1, 0:-2]                # Bring left neighbour to the current node
        sRi = sPad[1:-1, 2:  ]                # Bring right neighbour to the current node

        # Energy cost
        return 2 * self.spins * ( self.Jx*(sLe+sRi) + self.Jy*(sUp+sDo) ) - 2 * self.spins * self.extH

        
    def thermalEvolution(self, T, n=-1):
        '''
        Update the spin lattice based on a Metropolis algorithm.
        The cost of flipping every spins is computed in a vectorized manner.
        A flipping decision is computed for each spin, based on energy gain
        or thermal fluctuations. Finally, n random nodes are selected 
        and they are then flipped or not, based on the precedently computed
        decision. By default, n is set to 10% of the spins.
        
        Modify in place the spin lattice, and return a copy of the modified 
        spin lattice.
        '''
                
        # Get the number of random sites to modify (50% of the lattice by default)
        if n == -1:
            n = int(self.spins.size * 0.5)
        elif n > self.spins.size:
            raise NameError('Number of sites to be changed must low than spins sites')
        
        # Choose random sites
        pts = np.random.randint(low=0, high=self.spins.shape[0], size=(n, 2))
        Xs, Ys = pts[:, 0], pts[:, 1]
    
        # Energy cost
        cost = self.cost()
        
        # Conditions to flip the spin
        flip  = (cost < 0) | (np.random.rand(*self.spins.shape) < np.exp(-cost/T))
    
        # Update the configuration of these sites
        self.spins[Xs, Ys] *= -1 * flip[Xs, Ys] + 1 * ~flip[Xs, Ys]

        # Return the spin lattice
        return self.spins.copy()
        
        
    def twoPtsCorr(self, T, nEvolution):
        '''
        Compute the correlation function as function of the distance:
            G(r) = < si-<si> > * <sj - <sj> >
        where r is the distance d(i, j) between spin i and spin j.
        The mean are computed over Ntime thermal evolutions. In practice,
        the spin in the middle of the lattice is taken as reference. What
        is computed is then:
           G(r) = < s0-<s0> > * <sj - <sj> >
            
        First a serie of nEvolution spin states are generated at temperature T
        leading to a (Ntime, N, N) array where the first dimension
        is the temporal evolution at thermal equilibrium.
            
        The state of the spin Lattice is not affected by the calculation.

        Return 2 1D-array of shape (N*N - 1) correponding to all pairs (0, j)
          - r (array of distance), g (array of correlations)
        '''

        # Save initial state
        initState = self.spins.copy()
        
        # Computing nEvolution states of the spin lattice
        a = np.array([self.thermalEvolution(T) for _ in np.arange(nEvolution)])

        # Setup the spin state as it was
        self.spins = initState
        
        # Take the center of the lattice
        N = self.spins.shape[1]
        x0, y0 = int(N/2), int(N/2)
        s0 = a[:, x0, y0]
        
        # Take all the other spins, excluding (x0, y0)-node.
        xj, yj = np.meshgrid(np.arange(N), np.arange(N))
        xj, yj = xj.flatten(), yj.flatten()
        noCenter = (xj!=x0) | (yj!=y0)
        xj, yj = xj[noCenter], yj[noCenter] 
        sj = a[:, xj, yj]
        
        # Compute the distances
        r = np.sqrt((xj-x0)**2 + (yj-y0)**2)
        
        # Compute mean-substracted (reduced) spins
        s0r = s0 - s0.mean()
        sjr = sj - np.mean(sj, axis=0)
        
        # Compute G(r)
        g = np.mean(sjr * s0r[:, np.newaxis], axis=0)
        
        # Sort points by increasing radius
        sort = np.argsort(r)

        # Return the radius and 2-points correlations.
        return r[sort], g[sort]


    def simulatePhaseTransition(self, Tmin=0.1, Tmax=5, nT=50, 
                                Nther=150, Nmeas=10, Nswitch=-1,
                                reset=False, random=False, revert=False):
    
        '''
        Run a metropolis simulation on tje spin lattice using
        Nther iterations for thermalization of the system and Nmeas 
        iterations to measure system properties. Those are done for each
        temperatures, scanning from Tmin to Tmax using nT steps.
    
        The state of the spin lattice is not modified by a simulation.

        Returning several 1D np.array as function of temperature:
          - temperature, energy, magnetization, entropy
        
        Options:
          - Nther   [default:   150]: number of thermal evolutions before measuring
          - Nmeas   [default:    10]: number of thermal evolution to average observables
          - Nswitch [default:   10%]: number of spins which are possibly flipped in one thermal evolution
          - reset   [default: false]: reset spin state for each temperature
          - random  [default: false]: random initial state, otherwise the
                                      lowest energy configuration is taken
          - revert [default: false]: scan temperatures from hottest to coldest.
        '''

        # Save initial state
        initState = self.spins.copy()
        
        # Initialize arrays
        Ts = np.linspace(Tmin, Tmax, nT)
        Es = np.zeros(nT)
        Ms = np.zeros(nT)
        Ss = np.zeros(nT)
        
        if revert:
            Ts = Ts[::-1]
            
        # Helper function to initialize the spin lattice
        def initState(rand):
            if rand:
                self.randomize()
            else:
                self.align()
    
        # Initial state as aligned spins - to avoid domains.
        initState(random)    

        # Loop over temperatures
        for iT, T in enumerate(Ts):
        
            # Container for this temperature
            e, m, s = np.zeros(Nmeas), np.zeros(Nmeas), np.zeros(Nmeas)
        
            # Reset state if the option is enabled
            if reset:
                initState(random)   
        
            # Thermalization
            for iTh in range(Nther):
                self.thermalEvolution(T, n=Nswitch)

            # Compute energies and magnetism
            for iM in range(Nmeas):
                self.thermalEvolution(T, n=Nswitch)
                e[iM] = self.energy()
                m[iM] = self.magnetization()
                s[iM] = self.entropy()

            # Avergage all measurements for this temperaturs
            Es[iT] = np.mean(e)
            Ms[iT] = np.mean(m)
            Ss[iT] = np.mean(s)

        # Setup the spin state as it was
        self.spins = initState

        # Return the results
        return Ts, Es, Ms, Ss


    def animate(self, T, nEvolutions=50, saveName='', interval=50):

        '''
        Create a animation of spin lattice evolution at a temperature
        T, for nEvolutions steps. The output video can be saved under
        'saveName' (if empty, the video is not saved).

        The state of the spin lattice is not modified by an animation.

        Return the animation. To be displayed in a notebook, do
        >>> from IPython.display import HTML
        >>> anim = s.animate(T, N)
        >>> HTML(anim.to_html5_video())
        '''

        initState = self.spins.copy()
        imgs = [self.thermalEvolution(T) for _ in range(nEvolutions)]
        self.spins = initState
        
        # Create the animation
        anim = animateImages(imgs, interval)
        
        # Set up formatting for the movie files
        if saveName:
            anim.save('{}.gif'.format(saveName), writer='imagemagick', fps=60)

        return anim


def animateImages(imgs, interval=50):

    '''
    Return an animation made of several images.
    imgs = list of 2D np.array.
    '''
    
    def init():
        img.set_data(imgs[0])
        return (img,)

    def animate(i):
        img.set_data(imgs[i])
        return (img,)

    fig = plt.figure()
    ax  = fig.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img = ax.imshow(imgs[0],  cmap='BuPu', origin='bottom')
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(imgs),
                                   interval=interval, blit=True, save_count=len(imgs))
    plt.tight_layout()
    
    return anim
