#-------------------------------------------------------------------------------
# Name:        FFT-MA Level-Sim (Layered Copula)
# Purpose:     Simulation of non-Gaussian spatial random fields
#
# Author:      Dr.-Ing. S. Hoerning
#
# Created:     01/07/2022, Centre for Natural Gas, EAIT,
#                          The University of Queensland, Brisbane, QLD, Australia
#-------------------------------------------------------------------------------

import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as st
from statsmodels.distributions.empirical_distribution import ECDF
from helper_func import covariancefunction as covfun
from helper_func import empspast_anisotropic
import level_sim_conditional
import gstools as gs
import pykrige

# SIMULATE
n_realisations = 10

for s in range(n_realisations):
	covmods = []
	nlev = 41 # has to be an odd number to get phi_(tau)=0
	tau = np.linspace(0.0, 1, nlev)
	r1 = 40
	r2 = 4
	for ii, r in enumerate(tau):
		r = (1- tau[ii])*r1 + tau[ii]*r2
		covmod =  '0.01 Nug(0.0) + 0.99 Exp({})'.format(r)
		covmods.append(covmod)
	print(covmods)

	# initialize FFTMA_LS with the covmodes defined above
	fftmals = level_sim_conditional.FFTMA_LS(domainsize=(500, 500), covmods=covmods)

	# read conditioning values
	xy = np.load('data/small_xy.npy')
	condfield = np.load('data/small_sample_field.npy')
	cv = condfield[xy[:,0], xy[:,1]]


	# OK simulation for tau0fields
	ok_cov = gs.Exponential(dim=2, var=1, len_scale=22, nugget=0.01)
	domainsize = (532, 532)
	gridx = np.arange(0.0, domainsize[0], 1)
	gridy = np.arange(0.0, domainsize[1], 1)

	ok_data = pykrige.OrdinaryKriging(xy[:,0], xy[:,1], cv, ok_cov)
	z_data, s_data = ok_data.execute("grid", gridx, gridy)
	z_data = z_data.data.T
	
	tau0fields = RM.finalFields

	# start conditional layer cop simulation
	Y = fftmals.condsim(xy, cv, tau0fields, nsteps=70, kbw=10)

	# plot the results
	sim_cv = Y[xy[:, 0], xy[:, 1]]
	dif = np.sum((cv - sim_cv) ** 2)
	print(dif)

	als = np.array([-3,3])
	plt.figure()
	plt.scatter(cv ,sim_cv)
	plt.plot(als,als)
	plt.title('sq diff = {}'.format(dif))
	plt.savefig(r'scatter_{}.png'.format(s))
	plt.clf()
	plt.close()

	plt.figure()
	plt.imshow(Y, interpolation='nearest', origin='lower', cmap='jet', vmin=-3.6, vmax=3.6)
	plt.colorbar()
	plt.savefig(r'csimfield_{}.png'.format(s))
	plt.clf()
	plt.close()

	np.save('csfield_{}.npy'.format(s), Y)