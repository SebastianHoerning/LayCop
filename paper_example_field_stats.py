#-------------------------------------------------------------------------------
# Name:        FFT-MA Level-Sim
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
from RMWSPy.rmwspy import *
from helpers import covariancefunction as covfun
from spastats import varioFFT, empspast_anisotropic
import level_sim


def paper_plot(field, statlist_biv_mean):#, d, avvl, avhl):

	fontsize = 10
	plt.rcParams['font.size'] = fontsize
	plt.rcParams['legend.fontsize'] = fontsize
	plt.rcParams['figure.subplot.bottom'] = 0.05
	plt.rcParams['figure.subplot.top'] = 0.95
	plt.rcParams['figure.subplot.left'] = 0.07
	plt.rcParams['figure.subplot.right'] = 0.94
	plt.rcParams['figure.subplot.wspace'] = 0.2

	fig = plt.figure(figsize=(13, 8))
	axfield = plt.subplot2grid((3,2), (0,0), rowspan=2, aspect='equal')
	plt.imshow( field,
				interpolation='nearest',
				origin='lower',
				cmap='jet',
				vmin=-3.5,
				vmax=3.5
				)
	plt.xlabel('x')
	plt.ylabel('y')
	cb = plt.colorbar(shrink=0.8)
	for t in cb.ax.get_yticklabels():
			t.set_fontsize(fontsize)

	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)

	# rank correlation function
	h = statlist_biv_mean['h']
	# R = statlist_biv_mean['R']
	v = statlist_biv_mean['variogram']
	# C = statlist_biv_mean['covariance']

	ax0 = plt.subplot2grid((3,2), (0,1))
	plt.plot(h, v, 'x-', color='black', label='$\gamma_{iso}(h)$', alpha=0.7)
	# plt.plot(d, avhl, ':', color='black', label='$\gamma_{major}(h)$', alpha=0.7)
	# plt.plot(d, avvl, '--', color='black', label='$\gamma_{minor}(h)$', alpha=0.7)
	plt.plot(statlist_biv_mean['h_major'], statlist_biv_mean['variogram_major'], ':', color='black', label='$\gamma_{major}(h)$', alpha=0.7)
	plt.plot(statlist_biv_mean['h_minor'], statlist_biv_mean['variogram_minor'], '--', color='black', label='$\gamma_{minor}(h)$', alpha=0.7)
	leg = plt.legend(loc=4, fancybox=True)
	leg.get_frame().set_alpha(0.8)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlim(0,)
	plt.yticks([0.0,0.5,1.0])
	plt.ylim(-0.1,1.1)
	plt.ylabel('Variogram', fontsize=fontsize)

	# ASYMMETRIES
	# A = statlist_biv_mean['A']
	Atn = statlist_biv_mean['A_t_normed']

	plt.subplot2grid((3,2), (1,1), sharex=ax0)
	# A_t normed by Amax
	plt.plot(h, Atn, '1-', color='blue',label='$A(h)$',linewidth=1.5)
	plt.plot([0,h.max()], [0,0], '--', color='blue',linewidth=0.5, alpha=0.6)
	plt.ylabel('Asymmetry', fontsize=fontsize)
	plt.ylim(-0.35,0.35)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlabel('distance')
	leg = plt.legend(loc=1, fancybox=True)
	leg.get_frame().set_alpha(0.8)

	# BIVARIATE COPULA DENSITIES ------------------------------------------#
	c = statlist_biv_mean['bivariate_copula'][[0,1,2,4,6]]
	cmax = 3
	grid = ImageGrid(   fig,
						313, # similar to subplot(111)
						nrows_ncols = (1, c.shape[0]), # creates grid of axes
						#grids = c.shape[0],    # number of grids
						axes_pad=0.05, # pad between axes in inch.
						share_all=True,
						label_mode = '1',
						)
	for i in range(c.shape[0]):
		im = grid[i].imshow( c[i],
							extent=(0.,1.,0.,1.),
							origin='lower',
							interpolation='nearest',
							vmin=0,
							vmax=cmax,
							cmap='coolwarm'
							)
	grid[0].set_xticks([0,0.5,1])
	grid[0].set_yticks([0,0.5,1])

	axins = inset_axes(grid[-1],
				width="8%", # width = 10% of parent_bbox width
				height="100%", # height : 50%
				loc=3,
				bbox_to_anchor=(1.05, 0., 1, 1),
				bbox_transform=grid[-1].transAxes,
				borderpad=0,
				)
	plt.colorbar(im, cax=axins, ticks=[1,2,3])

	return fig


### START EXAMPLES
seed = 819074
# grid for spast for all examples
xyz = np.mgrid[[slice(0, 1000, 1) for i in range(2)]].reshape(2, -1).T
np.random.shuffle(xyz)
xyz = xyz[:30000]

# lagbounds for spast
lb = np.array([  0, 5, 10, 20,  40,  60,  80, 100, 120, 150, 180, 220, 250])

# base case Gaussian field
covmods =  ['0.01 Nug(0.0) + 0.99 Exp(40.0)']
np.random.seed(seed)
fftmals = level_sim.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods,)# anisotropies=[(1, 0.2, 25)])
field = fftmals.simnew()
rfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
Y1 = st.norm.ppf(rfield)

plt.figure()
plt.imshow(Y1, interpolation='nearest', cmap='jet', vmin=-3.6, vmax=3.6)
plt.colorbar()
plt.savefig(r'ex_basecase.png', dpi=250)
plt.clf()
plt.close()

# spast
vals = Y1[xyz[:,0], xyz[:,1]]
spast = empspast_anisotropic.empspast_isotropic_unstructured(xyz, vals)
spast.calc_stats(lagbounds=lb, ang_bounds_maj=[55, 75], ang_bounds_min=[145, 165])
spast.plt_stats()
plt.savefig('ex_basecase_statsall.png', dpi=300)
plt.clf()
plt.close()

# spast from variofft
vfft = varioFFT.varioFFT2D(Y1, icode=1)

ws = 250
w = [int(vfft.shape[i]/2) for i in range(2)]
gh = vfft[w[0] - ws : w[0] + ws, w[1] - ws : w[1] + ws]

# plt.imshow(gh, cmap='jet', origin='lower')
# plt.colorbar()
# plt.show()

vl = gh[250, 250:500]
hl = gh[250:500, 250]
dl45 = gh[250: , 250:]
dl45 = np.diag(dl45)
dl135 = gh[:250 , 250:][::-1]
dl135 = np.diag(dl135)

# average ll
avvl = []
avhl = []
avdl45 = []
avdl135 = []
for i in range(21):
	avvl.append(np.mean(vl[i*10 : (i+1)*10]))
	avhl.append(np.mean(hl[i*10 : (i+1)*10]))
	avdl45.append(np.mean(dl45[i*10 : (i+1)*10]))
	avdl135.append(np.mean(dl135[i*10 : (i+1)*10]))
avvl = np.array(avvl)
avhl = np.array(avhl)
avdl45 = np.array(avdl45)
avdl135 = np.array(avdl135)
d = np.arange(5, avvl.shape[0]*10, 10)

# plt.figure()
# plt.plot(d, avvl, 'x', c='black')
# plt.plot(d, avvl, c='black', linewidth=2.5)
# plt.plot(d, avhl, 'x', c='green')
# plt.plot(d, avhl, c='green', linewidth=2.5)
# plt.plot(d, avdl45, 'x', c='blue')
# plt.plot(d, avdl45, c='blue', linewidth=2.5)
# plt.plot(d, avdl135, 'x', c='pink')
# plt.plot(d, avdl135, c='pink', linewidth=2.5)
# plt.plot(spast.statlist_biv[0]['h'], spast.statlist_biv[0]['variogram'], 'o', c='red')
# plt.plot(spast.statlist_biv[0]['h'], spast.statlist_biv[0]['variogram'], c='red', linewidth=2.5)
# # plt.hlines(0, 0, d.max(), linestyles='--', colors='grey', alpha=0.6, lw=0.6)
# plt.yticks([0.0,0.5,1.0])
# plt.ylim(-0.1,1.0)
# plt.show()

paper_plot(Y1, spast.statlist_biv[0])#, d[:-3], avvl[:-3], avhl[:-3])
plt.savefig('ex_basecase_stats.png', dpi=250)
plt.clf()
plt.close()


# example 1: linearly changing Exponential variogram with range from 5 to 80
covmods = []
nlev = 40
ranges = np.linspace(0.0, 1, nlev)[::-1]
r1 = 5
r2 = 80
for ii, r in enumerate(ranges):
	r = (1- ranges[ii])*r1 + ranges[ii]*r2
	covmod =  '0.01 Nug(0.0) + 0.99 Exp({})'.format(r)
	covmods.append(covmod)
print(covmods)

np.random.seed(seed)
fftmals = level_sim.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods, reverse=True)
field = fftmals.simnewls()
rfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
Y1 = st.norm.ppf(rfield)

plt.figure()
plt.imshow(Y1, origin='lower', interpolation='nearest', cmap='jet', vmin=-3.6, vmax=3.6)
plt.colorbar()
plt.savefig(r'ex1_linear.png', dpi=250)
plt.clf()
plt.close()

# spast
vals = Y1[xyz[:,0], xyz[:,1]]
spast = empspast_anisotropic.empspast_isotropic_unstructured(xyz, vals)
spast.calc_stats(lagbounds=lb, ang_bounds_maj=[55, 75], ang_bounds_min=[145, 165])
spast.plt_stats()
plt.savefig('ex1_statsall.png', dpi=300)
plt.clf()
plt.close()

# spast from variofft
vfft = varioFFT.varioFFT2D(Y1, icode=1)

ws = 250
w = [int(vfft.shape[i]/2) for i in range(2)]
gh = vfft[w[0] - ws : w[0] + ws, w[1] - ws : w[1] + ws]

# plt.imshow(gh, cmap='jet', origin='lower')
# plt.colorbar()
# plt.show()

vl = gh[250, 250:500]
hl = gh[250:500, 250]
dl45 = gh[250: , 250:]
dl45 = np.diag(dl45)
dl135 = gh[:250 , 250:][::-1]
dl135 = np.diag(dl135)

# average ll
avvl = []
avhl = []
avdl45 = []
avdl135 = []
for i in range(21):
	avvl.append(np.mean(vl[i*10 : (i+1)*10]))
	avhl.append(np.mean(hl[i*10 : (i+1)*10]))
	avdl45.append(np.mean(dl45[i*10 : (i+1)*10]))
	avdl135.append(np.mean(dl135[i*10 : (i+1)*10]))
avvl = np.array(avvl)
avhl = np.array(avhl)
avdl45 = np.array(avdl45)
avdl135 = np.array(avdl135)
d = np.arange(5, avvl.shape[0]*10, 10)

paper_plot(Y1, spast.statlist_biv[0])#, d[:-3], avvl[:-3], avhl[:-3])
plt.savefig('ex1_stats.png', dpi=250)
plt.clf()
plt.close()


# example 2: Linearly changing relative nugget from 0.5 to 0.0 with exponential
# variogram with range 50
covmods = []
nlev = 40
ranges = np.linspace(0.0, 1, nlev)[::-1]
n1 = 0.5
n2 = 0.0
for ii, r in enumerate(ranges):
	nug = (1- ranges[ii])*n1 + ranges[ii]*n2
	covmod =  '{} Nug(0.0) + {} Exp(50.0)'.format(nug, 1 - nug)
	covmods.append(covmod)
print(covmods)

np.random.seed(seed)
fftmals = level_sim.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods, reverse=True)
field = fftmals.simnewls()
rfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
Y1 = st.norm.ppf(rfield)

plt.figure()
plt.imshow(Y1, origin='lower', interpolation='nearest', cmap='jet', vmin=-3.6, vmax=3.6)
plt.colorbar()
plt.savefig(r'ex2_linear.png', dpi=250)
plt.clf()
plt.close()

# spast
vals = Y1[xyz[:,0], xyz[:,1]]
spast = empspast_anisotropic.empspast_isotropic_unstructured(xyz, vals)
spast.calc_stats(lagbounds=lb, ang_bounds_maj=[55, 75], ang_bounds_min=[145, 165])
spast.plt_stats()
plt.savefig('ex2_statsall.png', dpi=300)
plt.clf()
plt.close()

# spast from variofft
vfft = varioFFT.varioFFT2D(Y1, icode=1)

ws = 250
w = [int(vfft.shape[i]/2) for i in range(2)]
gh = vfft[w[0] - ws : w[0] + ws, w[1] - ws : w[1] + ws]

vl = gh[250, 250:500]
hl = gh[250:500, 250]
dl45 = gh[250: , 250:]
dl45 = np.diag(dl45)
dl135 = gh[:250 , 250:][::-1]
dl135 = np.diag(dl135)

# average ll
avvl = []
avhl = []
avdl45 = []
avdl135 = []
for i in range(21):
	avvl.append(np.mean(vl[i*10 : (i+1)*10]))
	avhl.append(np.mean(hl[i*10 : (i+1)*10]))
	avdl45.append(np.mean(dl45[i*10 : (i+1)*10]))
	avdl135.append(np.mean(dl135[i*10 : (i+1)*10]))
avvl = np.array(avvl)
avhl = np.array(avhl)
avdl45 = np.array(avdl45)
avdl135 = np.array(avdl135)
d = np.arange(5, avvl.shape[0]*10, 10)

paper_plot(Y1, spast.statlist_biv[0])#, d[:-3], avvl[:-3], avhl[:-3])
plt.savefig('ex2_stats.png', dpi=250)
plt.clf()
plt.close()


# example 3: Linearly changing relative nugget from 0.5 to 0.0 with exponential
# variogram with range changing from 5 to 80
covmods = []
nlev = 40
ranges = np.linspace(0.0, 1, nlev)[::-1]
r1 = 5
r2 = 80
n1 = 0.5
n2 = 0.0
for ii, r in enumerate(ranges):
	r = (1- ranges[ii])*r1 + ranges[ii]*r2
	nug = (1- ranges[ii])*n1 + ranges[ii]*n2
	covmod =  '{} Nug(0.0) + {} Exp({})'.format(nug, 1 - nug, r)
	covmods.append(covmod)
print(covmods)

np.random.seed(seed)
fftmals = level_sim.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods, reverse=True)
field = fftmals.simnewls()
rfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
Y1 = st.norm.ppf(rfield)

plt.figure()
plt.imshow(Y1, origin='lower', interpolation='nearest', cmap='jet', vmin=-3.6, vmax=3.6)
plt.colorbar()
plt.savefig(r'ex3_linear.png', dpi=250)
plt.clf()
plt.close()

# spast
vals = Y1[xyz[:,0], xyz[:,1]]
spast = empspast_anisotropic.empspast_isotropic_unstructured(xyz, vals)
spast.calc_stats(lagbounds=lb, ang_bounds_maj=[55, 75], ang_bounds_min=[145, 165])
spast.plt_stats()
plt.savefig('ex3_statsall.png', dpi=300)
plt.clf()
plt.close()

# spast from variofft
vfft = varioFFT.varioFFT2D(Y1, icode=1)

ws = 250
w = [int(vfft.shape[i]/2) for i in range(2)]
gh = vfft[w[0] - ws : w[0] + ws, w[1] - ws : w[1] + ws]

vl = gh[250, 250:500]
hl = gh[250:500, 250]
dl45 = gh[250: , 250:]
dl45 = np.diag(dl45)
dl135 = gh[:250 , 250:][::-1]
dl135 = np.diag(dl135)

# average ll
avvl = []
avhl = []
avdl45 = []
avdl135 = []
for i in range(21):
	avvl.append(np.mean(vl[i*10 : (i+1)*10]))
	avhl.append(np.mean(hl[i*10 : (i+1)*10]))
	avdl45.append(np.mean(dl45[i*10 : (i+1)*10]))
	avdl135.append(np.mean(dl135[i*10 : (i+1)*10]))
avvl = np.array(avvl)
avhl = np.array(avhl)
avdl45 = np.array(avdl45)
avdl135 = np.array(avdl135)
d = np.arange(5, avvl.shape[0]*10, 10)

paper_plot(Y1, spast.statlist_biv[0])#, d[:-3], avvl[:-3], avhl[:-3])
plt.savefig('ex3_stats.png', dpi=250)
plt.clf()
plt.close()


# example 4: Linearly changing exponential variogram with range from 20 to 80 and anisotropy in either
# high or low values
for c in range(4):
	if c == 0:
		# case 1: low vals with low range and ani
		covmods = []
		anisotropies = []
		nlev = 40
		ranges = np.linspace(0.0, 1, nlev)#[::-1]
		r1 = 80
		r2 = 5
		a1 = 1
		a2 = 0.2
		for ii, r in enumerate(ranges):
			r = (1- ranges[ii])*r1 + ranges[ii]*r2
			af = (1- ranges[ii])*a1 + ranges[ii]*a2
			ani = (1, af, 25)
			covmod =  '0.01 Nug(0.0) + 0.99 Exp({})'.format(r)
			covmods.append(covmod)
			anisotropies.append(ani)
		print(covmods)

		np.random.seed(seed)
		fftmals = level_sim.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods, anisotropies=anisotropies, reverse=True)
		field = fftmals.simnewls()
		rfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
		Y1 = st.norm.ppf(rfield)

	elif c == 1:
		# case 2: high vals with high range and ani
		covmods = []
		anisotropies = []
		nlev = 40
		ranges = np.linspace(0.0, 1, nlev)#[::-1]
		r1 = 80
		r2 = 5
		a1 = 0.2
		a2 = 1
		for ii, r in enumerate(ranges):
			r = (1- ranges[ii])*r1 + ranges[ii]*r2
			af = (1- ranges[ii])*a1 + ranges[ii]*a2
			ani = (1, af, 25)
			covmod =  '0.01 Nug(0.0) + 0.99 Exp({})'.format(r)
			covmods.append(covmod)
			anisotropies.append(ani)
		print(covmods)

		np.random.seed(seed)
		fftmals = level_sim.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods, anisotropies=anisotropies, reverse=True)
		field = fftmals.simnewls()
		rfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
		Y1 = st.norm.ppf(rfield)

	elif c == 2:
		# case 3: low vals with high range and ani
		covmods = []
		anisotropies = []
		nlev = 40
		ranges = np.linspace(0.0, 1, nlev)#[::-1]
		r1 = 80
		r2 = 5
		a1 = 0.2
		a2 = 1
		for ii, r in enumerate(ranges):
			r = (1- ranges[ii])*r1 + ranges[ii]*r2
			af = (1- ranges[ii])*a1 + ranges[ii]*a2
			ani = (1, af, 25)
			covmod =  '0.01 Nug(0.0) + 0.99 Exp({})'.format(r)
			covmods.append(covmod)
			anisotropies.append(ani)
		print(covmods)

		np.random.seed(seed)
		fftmals = level_sim.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods, anisotropies=anisotropies)# reverse=True)
		field = fftmals.simnewls()
		rfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
		Y1 = st.norm.ppf(rfield)

	elif c == 3:
		# case 4: overall ani same range
		covmods = []
		anisotropies = []
		nlev = 40
		ranges = np.linspace(0.0, 1, nlev)#[::-1]
		a1 = 1
		a2 = 0.2
		for ii, r in enumerate(ranges):
			
			af = (1- ranges[ii])*a1 + ranges[ii]*a2
			ani = (1, af, 25)
			covmod =  '0.01 Nug(0.0) + 0.99 Exp(50.)'
			covmods.append(covmod)
			anisotropies.append(ani)
		print(covmods)

		np.random.seed(seed)
		fftmals = level_sim.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods, anisotropies=anisotropies, reverse=True)
		field = fftmals.simnewls()
		rfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
		Y1 = st.norm.ppf(rfield)

	plt.figure()
	plt.imshow(Y1, origin='lower', interpolation='nearest', cmap='jet', vmin=-3.6, vmax=3.6)
	plt.colorbar()
	plt.savefig(r'ex4_linear_{}.png'.format(c), dpi=250)
	plt.clf()
	plt.close()

	# # indicator??
	# Y1 = np.where(Y1 < 0, Y1, np.nan)

	# spast
	vals = Y1[xyz[:,0], xyz[:,1]]
	spast = empspast_anisotropic.empspast_isotropic_unstructured(xyz, vals)
	spast.calc_stats(lagbounds=lb, ang_bounds_maj=[55, 75], ang_bounds_min=[145, 165])
	spast.plt_stats()
	plt.savefig('ex4_statsall_{}.png'.format(c), dpi=300)
	plt.clf()
	plt.close()

	# spast from variofft
	vfft = varioFFT.varioFFT2D(Y1, icode=1)

	# rotate for major minor axes
	vfft = scipy.ndimage.rotate(vfft, -25, axes=[0,1], reshape=False)

	ws = 250
	w = [int(vfft.shape[i]/2) for i in range(2)]
	gh = vfft[w[0] - ws : w[0] + ws, w[1] - ws : w[1] + ws]

	vl = gh[250, 250:500]
	hl = gh[250:500, 250]
	dl45 = gh[250: , 250:]
	dl45 = np.diag(dl45)
	dl135 = gh[:250 , 250:][::-1]
	dl135 = np.diag(dl135)

	# average ll
	avvl = []
	avhl = []
	avdl45 = []
	avdl135 = []
	for i in range(21):
		avvl.append(np.mean(vl[i*10 : (i+1)*10]))
		avhl.append(np.mean(hl[i*10 : (i+1)*10]))
		avdl45.append(np.mean(dl45[i*10 : (i+1)*10]))
		avdl135.append(np.mean(dl135[i*10 : (i+1)*10]))
	avvl = np.array(avvl)
	avhl = np.array(avhl)
	avdl45 = np.array(avdl45)
	avdl135 = np.array(avdl135)
	d = np.arange(5, avvl.shape[0]*10, 10)

	paper_plot(Y1, spast.statlist_biv[0])#, d[:-3], avvl[:-3], avhl[:-3])
	plt.savefig('ex4_stats_{}.png'.format(c), dpi=250)
	plt.clf()
	plt.close()


# example 5: Linearly changing exponential variogram with range from 20 to 80 and anisotropy in either
# high and low values
covmods = []
anisotropies = []
nlev = 40
ranges = np.linspace(0.0, 1, nlev)#[::-1]
r1 = 80
r2 = 5
a1 = 0.5
a2 = 0.1
alp1 = 25
alp2 = 70
for ii, r in enumerate(ranges):
	r = (1- ranges[ii])*r1 + ranges[ii]*r2
	af = (1- ranges[ii])*a1 + ranges[ii]*a2
	alp = (1- ranges[ii])*alp1 + ranges[ii]*alp2
	# if ii < nlev/2:
	# 	ani = (1, af, 25)
	# else:
	# 	ani = (1, af, 70)
	ani = (1, af, alp)
	covmod =  '0.01 Nug(0.0) + 0.99 Exp({})'.format(r)
	covmods.append(covmod)
	anisotropies.append(ani)
print(covmods)

np.random.seed(seed)
fftmals = level_sim.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods, anisotropies=anisotropies, reverse=True)
field = fftmals.simnewls()
rfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
Y1 = st.norm.ppf(rfield)

plt.figure()
plt.imshow(Y1, origin='lower', interpolation='nearest', cmap='jet', vmin=-3.6, vmax=3.6)
plt.colorbar()
plt.savefig(r'ex5_linear2.png', dpi=250)
plt.clf()
plt.close()

# spast
vals = Y1[xyz[:,0], xyz[:,1]]

spast = empspast_anisotropic.empspast_isotropic_unstructured(xyz, vals)
spast.calc_stats(lagbounds=lb, ang_bounds_maj=[55, 75], ang_bounds_min=[145, 165])
spast.plt_stats()
plt.savefig('ex5_statsall2.png', dpi=300)
plt.clf()
plt.close()

# spast from variofft
vfft = varioFFT.varioFFT2D(Y1, icode=1)
# rotate for major minor axes
vfft = scipy.ndimage.rotate(vfft, -25, axes=[0,1], reshape=False)
ws = 250
w = [int(vfft.shape[i]/2) for i in range(2)]
gh = vfft[w[0] - ws : w[0] + ws, w[1] - ws : w[1] + ws]

vl = gh[250, 250:500]
hl = gh[250:500, 250]
dl45 = gh[250: , 250:]
dl45 = np.diag(dl45)
dl135 = gh[:250 , 250:][::-1]
dl135 = np.diag(dl135)

# average ll
avvl = []
avhl = []
avdl45 = []
avdl135 = []
for i in range(21):
	avvl.append(np.mean(vl[i*10 : (i+1)*10]))
	avhl.append(np.mean(hl[i*10 : (i+1)*10]))
	avdl45.append(np.mean(dl45[i*10 : (i+1)*10]))
	avdl135.append(np.mean(dl135[i*10 : (i+1)*10]))
avvl = np.array(avvl)
avhl = np.array(avhl)
avdl45 = np.array(avdl45)
avdl135 = np.array(avdl135)
d = np.arange(5, avvl.shape[0]*10, 10)

paper_plot(Y1, spast.statlist_biv[0])#, d[:-3], avvl[:-3], avhl[:-3])
plt.savefig('ex5_stats2.png', dpi=250)
plt.clf()
plt.close()


# example 6: Linearly changing variogram model with range from 20 to 80 
covmods = []
anisotropies = []
nlev = 40
ranges = np.linspace(0.01, 0.99, nlev)[::-1]
nugg = 0.01
for ii, r in enumerate(ranges):
	
	covmod =  '{} Nug(0.0) + {} Gau(15) + {} Exp(80)'.format(nugg, 1 - ranges[ii] - nugg/2, ranges[ii] - nugg/2)
	covmods.append(covmod)
	
print(covmods)

np.random.seed(seed)
fftmals = level_sim.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods, reverse=True)
field = fftmals.simnewls()
rfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
Y1 = st.norm.ppf(rfield)

plt.figure()
plt.imshow(Y1, origin='lower', interpolation='nearest', cmap='jet', vmin=-3.6, vmax=3.6)
plt.colorbar()
plt.savefig(r'ex6_linear_2.png', dpi=250)
plt.clf()
plt.close()

# spast
vals = Y1[xyz[:,0], xyz[:,1]]

spast = empspast_anisotropic.empspast_isotropic_unstructured(xyz, vals)
spast.calc_stats(lagbounds=lb, ang_bounds_maj=[55, 75], ang_bounds_min=[145, 165])
spast.plt_stats()
plt.savefig('ex6_statsall_2.png', dpi=300)
plt.clf()
plt.close()

# spast from variofft
vfft = varioFFT.varioFFT2D(Y1, icode=1)

ws = 250
w = [int(vfft.shape[i]/2) for i in range(2)]
gh = vfft[w[0] - ws : w[0] + ws, w[1] - ws : w[1] + ws]

vl = gh[250, 250:500]
hl = gh[250:500, 250]
dl45 = gh[250: , 250:]
dl45 = np.diag(dl45)
dl135 = gh[:250 , 250:][::-1]
dl135 = np.diag(dl135)

# average ll
avvl = []
avhl = []
avdl45 = []
avdl135 = []
for i in range(21):
	avvl.append(np.mean(vl[i*10 : (i+1)*10]))
	avhl.append(np.mean(hl[i*10 : (i+1)*10]))
	avdl45.append(np.mean(dl45[i*10 : (i+1)*10]))
	avdl135.append(np.mean(dl135[i*10 : (i+1)*10]))
avvl = np.array(avvl)
avhl = np.array(avhl)
avdl45 = np.array(avdl45)
avdl135 = np.array(avdl135)
d = np.arange(5, avvl.shape[0]*10, 10)

paper_plot(Y1, spast.statlist_biv[0])#, d[:-3], avvl[:-3], avhl[:-3])
plt.savefig('ex6_stats_2.png', dpi=250)
plt.clf()
plt.close()