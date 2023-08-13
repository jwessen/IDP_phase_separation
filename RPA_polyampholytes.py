###########################################################################
# This code analyzes the trajectory files produced by the accompanying    #
# field theory simulation (FTS) code "FTS_trajectories_MPI.py". For each  #
# temperature/Bjerrum length, the thermal average of the chemical         #
# potential (mu) and osmotic pressure (Pi) are computed as a function of  #
# polymer bead density, and written to a file ending with "_mu_Pi.txt".   #
# The self-intersection points of these (mu(rho),Pi(rho)) curves are then #
# numerically found, which are used to construct the binodal curves that  #
# constitute the FTS phase diagrams.                                      #
#                                                                         #
# The code is used in the publication                                     #
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
# and follows the methods described therein.                              #
###########################################################################

import numpy as np
import scipy.optimize as opt

class PolySol_RPA:
    def __init__( self, seq, lB, v, a ):
        self.lB    = lB                 # Reduced Bjerrum length 
        self.seq   = np.array(seq)      # Charge sequence
        self.N     = self.seq.shape[0]  # Length of polyampholyte
        self.v     = v                  # Excluded volume parameter
        self.a     = a                  # Smearing length

        if np.sum(self.seq) != 0 :
            print('Error: the polymer must be charge neutral.')
            return -1 

        epsilon = 1e-7
        n_points = int(1e4)
        self.k = np.linspace( epsilon , 20. , n_points )
        self.dk = self.k[1]-self.k[0]

        self.gcc  = np.zeros(n_points)
        self.gmc  = np.zeros(n_points)
        self.gmm  = np.zeros(n_points)

        for alpha in range(0,self.N):
            for beta in range(0,self.N):
                self.gcc  += self.seq[alpha] * self.seq[beta] * np.exp( -np.abs(alpha-beta) * self.k**2 / 6. )
                self.gmc  += self.seq[alpha] *                  np.exp( -np.abs(alpha-beta) * self.k**2 / 6. )
                self.gmm  +=                                    np.exp( -np.abs(alpha-beta) * self.k**2 / 6. )

        self.gmm /= self.N
        self.gmc /= self.N
        self.gcc /= self.N

        self.set_v_lB( v, lB )

    # Sets the values of the excluded volume barameter v and the Bjerrum length lB, and computes the A & B 
    # factors used in the RPA integrals
    def set_v_lB(self, v, lB):
        self.v  = v
        self.lB = lB

        Gamma_tilde_squared = np.exp(-self.a**2 * self.k**2)
        self.A = Gamma_tilde_squared * ( self.v * self.gmm + 4.*np.pi*self.lB/self.k**2 * self.gcc )
        self.B = Gamma_tilde_squared**2 * 4.*np.pi*self.lB / self.k**2 * self.v * ( self.gmm*self.gcc - self.gmc**2 )

    # Returns chemical potential and osmotic pressure computed in RPA at polymer bead density rho = n * N / V
    def calc_mu_Pi(self, rho):
        # chemical potential
        mu_MFT = np.log(rho/self.N) + self.v*self.N*rho
        mu_integrand = self.k**2 * ( self.A + 2.*self.B*rho ) / (1. + self.A*rho + self.B*rho**2 )
        mu_fl = self.N * np.sum( mu_integrand ) * self.dk / ( 4. * np.pi**2 )    
        mu_RPA = mu_MFT + mu_fl

        # osmotic pressure
        p_MFT  = rho/self.N + self.v*rho**2/2.
        p_integrand = self.k**2 * ( ( self.A*rho + 2.*self.B*rho**2 ) / ( 1. + self.A*rho + self.B*rho**2 ) - np.log( 1. + self.A*rho + self.B*rho**2 )  )
        p_fl = np.sum( p_integrand ) * self.dk / ( 4. * np.pi**2 )
        p_RPA = p_MFT + p_fl

        return mu_RPA, p_RPA

    # This function returns the difference in chemical potential and osmotic pressure between
    # two polymer bead densities rho_L and rho_H. These are contained in the input variable x as:
    # x = [ log(rho_L), log(rho_H ) ]
    def diff_function(self, x):
        rho_L, rho_H = np.exp( x )

        mu_L, p_L = self.calc_mu_Pi(rho_L)
        mu_H, p_H = self.calc_mu_Pi(rho_H)

        p_diff  = p_L - p_H
        mu_diff = mu_L - mu_H

        return [ p_diff , mu_diff ]

    # This function tries to find the polymer beads densities of two co-existing bulk phases, given 
    # rho_L_in and rho_H_in as initial guesses. 
    def find_binodal(self, rho_L_in, rho_H_in):
        rho = opt.fsolve( self.diff_function , [np.log( rho_L_in ) , np.log( rho_H_in ) ]  )
        
        final_diff = self.diff_function(rho)
        rho_L, rho_H = np.sort( np.exp(rho) )

        if np.abs(rho_L - rho_H) <= 1e-5:
            print("Discarding solution, bulk densities too similar!")
            return np.array([-1])

        if np.sum(np.abs(final_diff)) >= 1e-5:
            print("Discarding solution, covergence not good enough!")
            return np.array([-1])

        return rho_L, rho_H



########### Usage example ###############
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import CL_seq_list as sl

    seq, _, _ = sl.get_the_charge("sv20")   # Charge sequence
    lB = 0.5                                # Reduced Bjerrum length
    v  = 0.0068                             # Excluded volume parameter
    a  = 1./np.sqrt(6.)                     # Smearing length

    PS = PolySol_RPA( seq , lB , v , a )    # Create the polymer solution object

    ###############################################################
    ######## Plot chemical potential vs osmotic pressure ##########
    ###############################################################
    rho = 10**np.linspace(-20,1,500)        # Polymer bead densities
    mu, Pi = np.array( [ PS.calc_mu_Pi(r) for r in rho ] ).T

    plt.plot(mu,Pi)
    plt.xlabel("Chemical potential")
    plt.ylabel("Osmotic pressure")
    plt.show()
    plt.close()

    ##############################################################
    ####### Calculate the co-extistence (binodal) curve ##########
    ##############################################################
    T_all = np.linspace(0.5,10,500) # temperatures T^*
    lB_all = 1./T_all

    # Initial guess of bulk densities for lowest temperature
    rho_L_in = 1e-30
    rho_H_in = 10

    rho_L = np.array([])
    rho_H = np.array([])
    lB_binodal = np.array([])

    for lB in lB_all:
        PS.set_v_lB(v,lB)
        rho = PS.find_binodal(rho_L_in, rho_H_in)
        if len(rho)==2:
            rho_L = np.append(rho_L,rho[0])
            rho_H = np.append(rho_H,rho[1])
            rho_L_in = rho[0]
            rho_H_in = rho[1]
            lB_binodal = np.append(lB_binodal, lB)

    # Critical point
    rho_c = (rho_L[-1] + rho_H[-1]) / 2.
    T_c = 1./lB_binodal[-1]

    # Binodal curve
    rho_binodal = np.concatenate( ( rho_L      , np.flip(rho_H)      ) )
    lB_binodal  = np.concatenate( ( lB_binodal , np.flip(lB_binodal) ) )
    T_binodal   = 1./lB_binodal

    # Plot RPA phase diagram
    plt.fill_between(rho_binodal,T_binodal, label='co-existance region')
    plt.plot(rho_binodal,T_binodal, color='black', label='binodal curve')
    plt.plot(rho_c, T_c,'*', color='C3', ms=10, label='critical point')
    plt.legend(frameon=False)
    plt.xlabel('Polymer bead density')
    plt.ylabel('Reduced temperature')
    plt.title('Phase diagram')
    plt.show()


