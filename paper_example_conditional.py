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
from helper_func import fftma
import level_sim_conditional
import gstools as gs

# SIMULATE
n_realisations = 2

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


	# OK for cv
	ok_cov = gs.Exponential(dim=2, var=1, len_scale=22, nugget=0.01)
	domainsize = (536, 536)
	mg = np.mgrid[[slice(0, domainsize[i], 1) for i in range(2)]].reshape(2,-1).T

	data_krig = gs.krige.Ordinary(ok_cov, [xy[:,0], xy[:,1]], cv, exact=True)
	z_data, s_data = data_krig([mg[:,0], mg[:,1]])
	z_data = z_data.reshape(domainsize)
	
	# FFTMA (without layer cop) for OK simulation for tau0fields
	fftma_ = fftma.FFTMA(domainsize=domainsize, covmod='0.01 Nug(0.0) + 0.99 Exp(22.0)')

	tau0fields = []
	for t in range(8):
		print('Kriging Simulation # {} '.format(t))
		rand_field = fftma_.simnew()
		cvrf = rand_field[xy[:,0], xy[:,1]]
		ok_rf = gs.krige.Ordinary(ok_cov, [xy[:,0], xy[:,1]], cvrf, exact=True)
		rf_data, ss = ok_rf([mg[:,0], mg[:,1]])
		rf_data = rf_data.reshape(domainsize)
		cfield = z_data + (rand_field - rf_data)
		tau0fields.append(cfield)
	tau0fields = np.array(tau0fields)


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