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


# SIMULATE
example100 = False	# change to True for the example with 100 observations
n_realisations = 10

for s in range(n_realisations):
	# example: linearly changing Exponential variogram with range from 5 to 80
	covmods = []
	nlev = 41 # has to be an odd number to get phi_(tau)=0
	tau = np.linspace(0.0, 1, nlev)
	r1 = 80
	r2 = 5
	for ii, r in enumerate(tau):
		r = (1- tau[ii])*r1 + tau[ii]*r2
		covmod =  '0.01 Nug(0.0) + 0.99 Exp({})'.format(r)
		covmods.append(covmod)
	print(covmods)

	# initialize FFTMA_LS with the covmodes defined above
	fftmals = FFTMA_LS(domainsize=(1000, 1000), covmods=covmods)

	# nsamp = 100
	# xy = np.ones([nsamp, 2])*99999
	# aa = np.zeros([1,2])
	# i = 0
	# while i < nsamp:
	# 	aa[0,:] = np.random.randint((850, 850), size=2) + [75, 75]
	# 	dst = sp.distance.cdist(xy, aa, metric='euclidean')
	# 	if np.min(dst) > 30:
	# 		xy[i,:] = aa[0,:]
	# 		i = i+1
	# xy = xy.astype(int)
	
	if example100:
		# EXAMPLE WITH 100 OBSERVATIONS
		# load conditioning point coords and values
		xy = np.load('xy_100.npy')
		condfield = np.load('sample_field.npy')
		cv = condfield[xy[:,0], xy[:,1]]
		# start RM
		RM = RMWS(domainsize=fftmals.sqrtFFTQ[0].shape, covmod=covmods[(nlev - 1)//2], nFields=10,
				cp=xy, cv=cv, norm=0.2)
		RM()
		tau0fields = RM.finalFields
		# np.save(r'tau0field_{}.npy'.format(s), tau0fields)

		# start conditional layer simulation
		# return the field and the corresponding random numbers
		Y = fftmals.condsim(xy, cv, tau0fields, nsteps=100, kbw=50)

	else:
		# EXAMPLE WITH 30 OBSERVATIONS
		# load conditioning point coords and values
		xy = np.load('xy.npy')
		condfield = np.load('sample_field.npy')
		cv = condfield[xy[:,0], xy[:,1]]
		# start RM
		RM = RMWS(domainsize=fftmals.sqrtFFTQ[0].shape, covmod=covmods[(nlev - 1)//2], nFields=5,
				cp=xy, cv=cv, norm=0.2)
		RM()
		tau0fields = RM.finalFields
		# np.save(r'tau0field_{}.npy'.format(s), tau0fields)

		# start conditional layer simulation
		# return the field and the corresponding random numbers
		Y = fftmals.condsim(xy, cv, tau0fields, nsteps=30, kbw=50)

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