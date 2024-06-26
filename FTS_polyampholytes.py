###########################################################################
# This code performs the Complex Langevin (CL) time evolution of a        #
# polymer solution of polyampholytes. The polymer is modelled as a        #
# sequence of charged beads, with a given charge sequence. The code       #
# uses the CL method to evolve the fields w and psi, which represent      #
# the polymer bead and charge density, respectively. The polymer          #
# solution is modelled in a cubic box of side length L, with Nx grid      #
# points in each direction. The code is based on the methods described    #
# in the publication:                                                     #
#                                                                         #
#    Y.-H. Lin, J. Wessén, T. Pal, S. Das and H.S. Chan (2022)            #
#    Numerical Techniques for Applications of Analytical Theories to      #
#    Sequence-Dependent Phase Separations of Intrinsically Disordered     #
#    Proteins.                                                            #
#    In: Phase-Separated Biomolecular Condensates, Methods and Protocols; #
#    Methods Mol. Biol;                                                   #
#    Zhou, H.-X., Spille, J.-H., Banerjee, P. R., Eds.;                   #
#    Springer-Nature, 2022; Vol. 2563, Chapter 3, pp 51−94,               #
#    DOI: 10.1007/978-1-0716-2663-4_3.                                    #
#    ( Pre-print available at arXiv:2201.01920 )                          #
#                                                                         #
# and follows the methods described therein. The main function shows      # 
# code's intended usage.                                                  #
###########################################################################

import numpy as np

from numpy.fft import fftn as ft
from numpy.fft import ifftn as ift

#----------------------- Define polymer solution as a class object -----------------------
class PolySol:
    def __init__( self, sigma, rhop0, lB, v, a, Nx=32, dx=None ):

        self.lB    = lB                  # Reduced Bjerrum length 
        self.sig   = np.array(sigma)     # Charge sequence
        self.N     = self.sig.shape[0]   # Length of polyampholyte
        self.rhop0 = rhop0               # Polymer bead bulk density, np*N/V 
        self.v     = v                   # Excluded volume parameter
        self.a     = a                   # Smearing length
        self.Nx    = Nx                  # Number of grid points: the resolution
        self.dx    = self.a if dx is None else dx # distance between two n.n. grid points
        self.L     = self.dx*Nx          # Box edge length
        self.V     = self.L**3           # Box volume
        self.dV    = self.dx**3          # delta volume of each grid 

        if np.sum(self.sig) != 0 :
            print('Error: the polymer must be charge neutral.')
            return -1 

        # wave number vectors of the grid space
        ks1d    = 2.*np.pi*np.fft.fftfreq(self.Nx,self.dx)              # k's in 1D reciprocal space
        self.kz = np.tile(ks1d, (self.Nx,self.Nx,1))                    # 3D array with kz[i,j,l] = ksld[l]
        self.kx = np.swapaxes( self.kz, 0, 2)                           # 3D array with kx[i,j,l] = ksld[i]
        self.ky = np.swapaxes( self.kz, 1, 2)                           # 3D array with ky[i,j,l] = ksld[j]
        self.k2 = self.kx*self.kx + self.ky*self.ky + self.kz*self.kz   # 3D array of k*k
         
        self.Gamma   = np.exp(-self.k2*self.a**2/2.) # Gaussian smearing
        self.Prop    = np.exp(-self.k2/6. )     	 # Gaussian chain n.n propagator

        self.GT2_w   = ( self.k2*self.a**2/3. - 1./2. )*self.Gamma  # smearing in pressure
        self.GT2_psi = ( self.k2*self.a**2/3. - 1./6. )*self.Gamma  # smearing in pressure

        # Gaussian chain correlation functions in the k-space
        Gij = np.exp(-np.tensordot(np.arange(self.N), self.k2, axes=0)/6)

        Mcc = np.kron(self.sig, self.sig).reshape((self.N, self.N))
        Tcc = np.array([ np.sum(Mcc.diagonal(n) + Mcc.diagonal(-n)) \
                         for n in range(self.N)]) 
        Tcc[0] /= 2

        Tmm = 2*np.arange(self.N,0,-1)
        Tmm[0] /= 2   
 
        Mmc = np.kron(self.sig, np.ones(self.N)).reshape((self.N, self.N))
        Tmc = np.array([ np.sum(Mmc.diagonal(n) + Mmc.diagonal(-n)) \
                         for n in range(self.N)])                
        Tmc[0] /= 2

        self.gcc = Gij.T.dot(Tcc).T / self.N
        self.gmm = Gij.T.dot(Tmm).T / self.N
        self.gmc = Gij.T.dot(Tmc).T / self.N

        # Fields
        self.w   = np.zeros( ( self.Nx, self.Nx, self.Nx ), dtype=complex )
        self.psi = np.zeros( ( self.Nx, self.Nx, self.Nx ), dtype=complex )

        # Single polymer partition function
        self.Q = 1

        # Chain propagators
        self.qF = np.zeros( ( self.N, self.Nx, self.Nx, self.Nx ), dtype=complex )
        self.qB = np.zeros( ( self.N, self.Nx, self.Nx, self.Nx ), dtype=complex )

        # Field operators for the bead- and charge density
        self.rhop = np.zeros( ( self.Nx, self.Nx, self.Nx ), dtype=complex ) + self.rhop0
        self.rhoc = np.zeros( ( self.Nx, self.Nx, self.Nx ), dtype=complex )

    # Taking Laplacian of x via Fourier transformation
    def lap(self, x):
        return -ift( self.k2 * ft( x ) ) 
    
    # Obtain densities from fields
    def calc_densities( self ):

        w_s   = ift( self.Gamma*ft( self.w )   )
        psi_s = ift( self.Gamma*ft( self.psi ) )    

        PSI =  1j*( np.tensordot( np.ones(self.N), w_s   , axes=0 ) + \
                    np.tensordot( self.sig       , psi_s , axes=0) )

        self.qF[0]  = np.exp( -PSI[0]  )
        self.qB[-1] = np.exp( -PSI[-1] )
    
        for i in range( self.N-1 ):
            # forwards propagator
            self.qF[i+1] = np.exp( -PSI[i+1] )*ift( self.Prop*ft(self.qF[i]) )
            # backwards propagator
            j = self.N-i-1
            self.qB[j-1] = np.exp( -PSI[j-1] )*ift( self.Prop*ft(self.qB[j]) )

        self.Q = np.sum( self.qF[-1] )  * self.dV / self.V
        qs = self.qF * self.qB*np.exp(PSI)

        self.rhop = self.rhop0 / self.N / self.Q * np.sum(qs, axis=0)
        self.rhoc = self.rhop0 / self.N / self.Q * qs.T.dot(self.sig).T 

    # Returns the polymer chemical potential for the current field configuration
    def get_chem_pot( self ):
        mu_p = np.log( self.rhop0 / self.N ) - np.log( self.Q ) 
        return mu_p

    # Returns the osmotic for the current field configuration
    def get_pressure( self ):
        ft_w   = ft( self.w )
        ft_psi = ft( self.psi )

        w_s   = ift( self.Gamma * ft_w   )
        psi_s = ift( self.Gamma * ft_psi )

        PSI = 1j*( np.tensordot( np.ones(self.N) , w_s   , axes=0 ) + \
                   np.tensordot( self.sig        , psi_s , axes=0 ) )

        qs = self.qB * self.qF * np.exp( PSI ) / self.Q

        lap_qB = np.array( [ self.lap( np.exp( PSI[i] ) * self.qB[i] ) for i in range(self.N)] )
        term1  = np.sum( self.qF * lap_qB ) / ( 9.*self.Q )

        term2  = 1j*np.sum( np.array([ qs[i] * ift( self.GT2_w * ft_w ) for i in range(self.N) ]) )
        term3  = 1j*np.sum( np.array([ qs[i] * self.sig[i] * ift( self.GT2_psi * ft_psi ) for i in range(self.N) ]) )

        Pi = (self.rhop0 / self.N) * (1.0 - ( term1 + term2 + term3 ) * self.dV / self.V)

        return Pi

    # Use this function to set/initialise the fields. 
    def set_fields(self, w, psi):
        self.w   = w
        self.psi = psi
        self.calc_densities()

    # Returns a dictionary containing all parameters, useful for printing to a file for archiving
    def get_params( self ):
        params = {
            "lB"    : self.lB    ,\
            "v"     : self.v     ,\
            "rhop0" : self.rhop0 ,\
            "N"     : self.N     ,\
            "V"     : self.V     ,\
            "L"     : self.L     ,\
            "Nx"    : self.Nx    ,\
            "dx"    : self.dx    ,\
            "dV"    : self.dV    ,\
            "a"     : self.a     ,\
        }
        return params

#---------------------------- Complex Langevin Time Evolution ----------------------------

# Performs one semi-implicit integration step of the CL evolution equations
def CL_step_SI(PS, M_inv, dt, useSI=True):

    std     = np.sqrt( 2 * dt / PS.dV )
    eta_w   = std*np.random.randn( PS.Nx, PS.Nx, PS.Nx ) 
    eta_psi = std*np.random.randn( PS.Nx, PS.Nx, PS.Nx ) 
 
    dw   = -dt*( 1j*ift( PS.Gamma*ft( PS.rhop ) ) + PS.w/PS.v                         ) + eta_w
    dpsi = -dt*( 1j*ift( PS.Gamma*ft( PS.rhoc ) ) - PS.lap(PS.psi) / (4.*np.pi*PS.lB) ) + eta_psi
    
    print(np.mean(np.abs(PS.w/PS.v)),np.mean(np.abs(- PS.lap(PS.psi) / (4.*np.pi*PS.lB))))

    if useSI: # Semi-implicit CL step
        ft_dw, ft_dpsi = ft( dw ) , ft( dpsi )
        dw_tmp   = M_inv[0,0] * ft_dw + M_inv[0,1] * ft_dpsi
        dpsi_tmp = M_inv[1,0] * ft_dw + M_inv[1,1] * ft_dpsi
 
        PS.w   += ift( dw_tmp ) 
        PS.psi += ift( dpsi_tmp ) 
    else:  # Euler CL step
        PS.w   += dw
        PS.psi += dpsi

    PS.calc_densities()
 
# get M_inv for semi-implicit CL integration method
def get_M_inv( PS, dt ):
    K11 = PS.Gamma**2 * PS.rhop0 * PS.gmm + 1. / PS.v
    K12 = PS.Gamma**2 * PS.rhop0 * PS.gmc
    K22 = PS.Gamma**2 * PS.rhop0 * PS.gcc + PS.k2 / (4.*np.pi*PS.lB)
    K11[0,0,0] = 1. / PS.v

    M = np.array( [ [ 1.+dt*K11 , dt*K12 ] , [ dt*K12 , 1.+dt*K22 ] ]  )
    det_M = M[0,0] * M[1,1] - M[0,1] * M[1,0]
    M_inv = np.array( [ [ M[1,1] , - M[0,1] ] , [ - M[1,0] , M[0,0] ] ] ) / det_M

    print(M_inv[0,0,:,0,0])


    return M_inv

#------------------------------------- Main function -------------------------------------

def av_density_to_1d(rho_in):
    rho = np.copy(rho_in)
    while len(rho.shape)>1:
        rho = np.mean(rho,axis=0)
    return rho


if __name__ == "__main__":

    #### for plotting #####
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = True
    #matplotlib.rcParams['text.latex.unicode'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 0.8 
    plt.rcParams['lines.linewidth'] = 3 
    ######################

    import CL_seq_list as sl
  
    dt = 0.01             # CL time step size
    t_prod = int(5e4)     # Total number of time steps

    lB  = 0.38            # Bjerrum length (inverse of the reduced temperature)
    v   = 0.0068          # Excluded volume parameter
    a   = 1./np.sqrt(6.)  # Smearing length
    rho = 0.1             # Polymer bead bulk density, n * N / V

    sig, N, the_seq = sl.get_the_charge("sv20")
    print(the_seq, N)
 
    # Polymer solution object
    PS = PolySol(sig, rho, lB, v, a , Nx=8)

    run_label = "data/example_FTS"

    # Save parameter values
    params = PS.get_params()
    with open( run_label + "_params.txt" , 'w') as f:
        print(params, file=f)

    # Initialize fields as random fluctuations around mean field solution
    init_size = 0.1
    w   = init_size * ( np.random.randn( PS.Nx,PS.Nx,PS.Nx ) + \
                   1j * np.random.randn( PS.Nx,PS.Nx,PS.Nx ) )
    psi = init_size * ( np.random.randn( PS.Nx,PS.Nx,PS.Nx ) + \
                   1j * np.random.randn( PS.Nx,PS.Nx,PS.Nx ) )

    w   -= np.mean(w) + 1j * rho * v
    psi -= np.mean(psi)
    PS.set_fields(w,psi)

    # Compute the M^(-1) matrix used for semi-implicit CL evolution
    Minv = get_M_inv( PS, dt)

    with open( run_label + "_traj.txt" , 'w') as f:
        for t in range(t_prod):
            if t %50 == 0: # Sample
                mu = PS.get_chem_pot()
                Pi = PS.get_pressure()

                f.write('{:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e}'.format(t, t*dt, mu.real , mu.imag, Pi.real, Pi.imag ) )                 
                f.write('\n')
                f.flush()

            # Take one CL time step
            CL_step_SI(PS, Minv, dt, useSI=True)

            if t%100==0:
                rhob = av_density_to_1d(PS.rhop)
                z = np.linspace(0,PS.L,PS.Nx) * 3.8
                plt.plot(z,rhob.real,'-' ,color='C0')
                plt.plot(z,rhob.imag,'--',color='C0')
                plt.title("t=" + str(t) + ", t="+str(t*dt))

                plt.show()
