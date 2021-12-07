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
#    Y.-H. Lin, J. Wessén, T. Pal, S. Das and H.S. Chan (2021)            #
#    Numerical Techniques for Applications of Analytical Theories to      #
#    Sequence-Dependent Phase Separations of Intrinsically Disordered     #
#    Proteins. Methods in Molecular Biology (Springer-Nature),            #
#    accepted for publication.                                            #
#                                                                         #
# and follows the methods described therein.                              #
###########################################################################

import numpy as np 
import scipy.optimize as opt
from scipy import interpolate
from scipy.signal import savgol_filter
from sympy.geometry import Point, Segment, intersection

# Set show_plot = True if you wish to monitor the search for
# co-existence points by plotting the (mu(rho),Pi(rho)) curves 
# along with the found intersection points.
show_plot = True
if show_plot:
    import matplotlib.pyplot as plt

###############################
######### Functions ###########
###############################

# Returns the differences in chemical potential and osmotic pressure for two
# densities contained in the array x = [ log(rho1), log(rho2) ]. This function
# is used for numerical solution of the phase equilibrium conditions.
def diff_func(x, rho, mu, Pi):
    rho_bulk = np.exp(x)
    rho_L = np.min(rho_bulk)
    rho_H = np.max(rho_bulk)

    if rho_L < rho[0] or rho_H > rho[-1]:
        return [1e5, 1e5]

    mu_int = interpolate.interp1d(rho,mu)
    Pi_int = interpolate.interp1d(rho,Pi)

    mu_diff = mu_int(rho_L) - mu_int(rho_H)
    Pi_diff = Pi_int(rho_L) - Pi_int(rho_H)

    return [ mu_diff , Pi_diff ]

def make_segment(Pi1,mu1,Pi2,mu2):
    p1,p2 = Point(Pi1, mu1, evaluate=False) , Point(Pi2, mu2, evaluate=False)
    return Segment(p1,p2, evaluate=False)

# Returns the bulk densities, chemical potential and osmotic pressure of the self-intersection point
# of the input {mu(rho), Pi(rho)} curve. If no satisfactory self-intersection point is found, the
# array [-1] is returned. 
def find_intersection(rho, mu, Pi):
    # Remove crashed runs (if any)
    indices = np.array([str(m)!='nan' for m in mu])
    rho = rho[indices]
    mu  = mu[indices]
    Pi  = Pi[indices]

    if show_plot:
        plt.plot(mu,Pi,'o',ms=4)
        plt.xlabel("chemical potential")
        plt.ylabel("osmotic pressure")

    # Slightly smooth the data. This makes the calculation more robust against
    # statistical fluctuations in the vicinity of the self-intersection point.
    mu = savgol_filter(mu, 11, 9) # window size 11, polynomial order 9
    Pi = savgol_filter(Pi, 11, 9)
    
    if show_plot:
        plt.plot(mu,Pi,'-')

    # The following looks for approximate solutions to mu(rho1)=mu(rho2), Pi(rho1)=Pi(rho2)
    # by turning the (mu(rho),Pi(rho)) curve into sympy.geometry Segments, and checking if 
    # each pair of segments intersect using the sympy intersection function. 
    print("Looking for approximate intersections...")
    segments = np.array([ make_segment( Pi[1],mu[1],Pi[0],mu[0] ) ,\
                          make_segment( Pi[2],mu[2],Pi[1],mu[1] ) ])
    rho_Hs = np.array([])
    rho_Ls = np.array([])
    intersection_found = False
    for i in range(3,len(rho)):
        if i%10==0: print(len(rho)-i)
        s1 = make_segment( Pi[i] , mu[i] , Pi[i-1] ,mu[i-1] ) 
        for j in range(len(segments)-1):
            s2 = segments[j]        
            intersec = s1.intersection(s2)
            if len(intersec) > 0:
                print("Intersection found.")
                intersection_found = True
                rho_Hs = np.append(rho_Hs, (rho[i]+rho[i-1])/2. )
                rho_Ls = np.append(rho_Ls, (rho[j]+rho[j-1])/2. )
        segments = np.append(segments,s1)
    
    if not intersection_found:
        print("Found no intersection.")
        if show_plot:
            plt.show()
            plt.close()
        return np.array([-1])

    # Each approximate intersection is now used as an initial guess for the scipy.optimization
    # function fsolve to find more accurate solutions to mu(rho1)=mu(rho2), Pi(rho1)=Pi(rho2).
    rho_H_final = np.array([])
    rho_L_final = np.array([])
    print("Using approximate intersections as inital guesses...")
    for i in range(len(rho_Hs)):
        rho_sol = opt.fsolve( diff_func , [np.log( rho_Ls[i] ) , np.log( rho_Hs[i] )] , args=(rho,mu,Pi) )
        final_diff = diff_func(rho_sol, rho,Pi,mu)
        rho_sol = np.exp(rho_sol)

        rho_L = np.min(rho_sol)
        rho_H = np.max(rho_sol)

        if np.abs(rho_L - rho_H) <= 1e-5:
            print("Discarding solution, bulk densities too similar!")
        elif np.sum(np.abs(final_diff)) >= 1e-5:
            print("Discarding solution, covergence not good enough!")
        else:
            rho_H_final = np.append(rho_H_final, rho_H )
            rho_L_final = np.append(rho_L_final, rho_L )

    if len(rho_H_final) == 0:
        print("All intersections were discarded.")
        if show_plot:
            plt.show()
            plt.close()
        return np.array([-1])

    mu_int = interpolate.interp1d(rho,mu)
    Pi_int = interpolate.interp1d(rho,Pi)

    mu_c = np.array( [mu_int(r) for r in rho_H_final] )
    Pi_c = np.array( [Pi_int(r) for r in rho_H_final] )

    # Plot all found self-intersection points.
    if show_plot:
        plt.plot(mu_c,Pi_c,'*', color='red', ms=8)

    # Unphysical self-intersection points are sometimes found around the spinodal turning
    # points (these tend to disappear with larger system sizes). The binodal self-intersection
    # points correspond to a larger difference between the high- and low density bulk phases. 
    # The solution with the largest density difference is therefore deemed the physical one. 
    # However, it is recommended visually verify this by plotting all found self-intersection points. 
    density_diffs = rho_H_final - rho_L_final
    index = np.argmax(density_diffs)

    rho_L_out = rho_L_final[index]
    rho_H_out = rho_H_final[index]
    mu_c_out  = mu_c[index]
    Pi_c_out  = Pi_c[index] 

    if show_plot:
        plt.plot(mu_c_out,Pi_c_out,'o', color='green', ms=10, fillstyle='none')
        plt.show()
        plt.close()

    return np.array( [ rho_L_out, rho_H_out, mu_c_out, Pi_c_out ])

##########################
######### Main ###########
##########################

ncpus = 40
all_rho = np.exp( np.linspace( np.log(1e-6), np.log(10), ncpus ) )  # All bead bulk densities considered. 
all_lB  = 1. / np.linspace(1.,6.,5)                                 # All values of the Bjerrum length (lB). 

t_equil = 50. # Equilibration Complex Langevin time 

run_label_base = "data/sv20"
with open(run_label_base + "_phase_diagram.txt", 'w') as pd_file:

    # Calculate thermally averaged chemical potentials and osmotic pressures from FTS trajectories
    for lB_ind in range(len(all_lB)):

        mu_av = np.zeros( ncpus , dtype=complex)
        Pi_av = np.zeros( ncpus , dtype=complex)
        for run in range(ncpus):
            # Read the trajectory file
            _, t, mu_R, mu_I, Pi_R, Pi_I = np.loadtxt( run_label_base+"_lB_"+str(lB_ind)+"_"+str(run)+"_traj.txt" ).T

            # Remove equilibration period
            prod_indices = t > t_equil
            mu_R = mu_R[ prod_indices ]
            mu_I = mu_I[ prod_indices ]
            Pi_R = Pi_R[ prod_indices ]
            Pi_I = Pi_I[ prod_indices ]

            # Compute Complex Langevin time averages
            mu_av[run] = np.mean( mu_R + 1j * mu_I )
            Pi_av[run] = np.mean( Pi_R + 1j * Pi_I )

        # Save mu(rho) and Pi(rho) to file. 
        out_data = np.array( [all_rho, mu_av.real, mu_av.imag, Pi_av.real, Pi_av.imag] )
        np.savetxt( run_label_base+"_lB_"+str(lB_ind)+"_mu_Pi.txt" , out_data.T )

        # Find bulk densities given by the self-intersection of the ( mu(rho), Pi(rho) ) curve
        sol = find_intersection(all_rho, mu_av.real, Pi_av.real )
        # Write binodal curve to file
        if len(sol) == 4:
            rho_L, rho_H , mu_c, Pi_c = sol

            pd_file.write('{:.8e} {:.8e} {:.8e} {:.8e} {:.8e}'.format(all_lB[lB_ind], rho_L, rho_H, mu_c, Pi_c) )                 
            pd_file.write('\n')
