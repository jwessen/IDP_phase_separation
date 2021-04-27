
#########################################################################
# Random phase approximation (RPA) code for calculating the chemical 	#
# potential and osmotic pressure of a system of polyampholytes. 		#
# This code is used in the publication									#
#																		#
#    Lin Y, Wessén J, Pal T, Das S & Chan H S, XXXXX (2021)  			#
#																		#
# and follows the notation therein. A usage example is provided in the 	#
# main function of this code.			 								#
#########################################################################

import numpy as np

class PolySol_RPA:
	def __init__( self, seq, lB, v, a ):
		self.lB    = lB 				# Reduced Bjerrum length 
		self.seq   = np.array(seq)		# Charge sequence
		self.N     = self.seq.shape[0]	# Length of polyampholyte
		self.v     = v 					# Excluded volume parameter
		self.a     = a   				# Smearing length

		if np.sum(self.seq) != 0 :
			print('Error: the polymer must be charge neutral.')
			return -1 

		epsilon = 1e-7
		n_points = int(1e4)
		self.k = np.linspace( epsilon , 20. , n_points )
		self.dk = self.k[1]-self.k[0]

		gcc  = np.zeros(n_points)
		gmc  = np.zeros(n_points)
		gmm  = np.zeros(n_points)

		for alpha in range(0,self.N):
			for beta in range(0,self.N):
				gcc  += self.seq[alpha] * self.seq[beta] * np.exp( -np.abs(alpha-beta) * self.k**2 / 6. )
				gmc  += self.seq[alpha] *                  np.exp( -np.abs(alpha-beta) * self.k**2 / 6. )
				gmm  +=                                    np.exp( -np.abs(alpha-beta) * self.k**2 / 6. )

		gmm /= self.N
		gmc /= self.N
		gcc /= self.N

		Gamma_tilde_squared = np.exp(-self.a**2 * self.k**2)
		self.A = Gamma_tilde_squared * ( self.v * gmm + 4.*np.pi*self.lB/self.k**2 * gcc )
		self.B = Gamma_tilde_squared**2 * 4.*np.pi*self.lB / self.k**2 * self.v * ( gmm*gcc - gmc**2 )

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


########### Usage example ###############
if __name__ == "__main__":

	import matplotlib.pyplot as plt
	import CL_seq_list as sl

	seq, _, _ = sl.get_the_charge("sv20")	# Charge sequence
	lB = 0.5								# Reduced Bjerrum length
	v  = 0.0068	 							# Excluded volume parameter
	a  = 1./np.sqrt(6.)						# Smearing length

	PS = PolySol_RPA( seq , lB , v , a )

	rho = 10**np.linspace(-20,1,500)
	mu, Pi = np.array( [ PS.calc_mu_Pi(r) for r in rho ] ).T

	plt.plot(mu,Pi)
	plt.xlabel("Chemical potential")
	plt.ylabel("Osmotic pressure")
	plt.show()
