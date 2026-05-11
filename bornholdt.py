# -*- coding: utf-8 -*-
from __future__ import division
# import matplotlib 
from pylab import *
# from scipy import *
import sys
import os
import glob
import random
# import scipy.optimize as optimize
# from scipy.optimize import curve_fit
import collections
import math
from numpy import linalg as LA
import torch

# def estimated_autocorrelation(x):
#     """
#     http://stackoverflow.com/q/14297012/190597
#     http://en.wikipedia.org/wiki/Autocorrelation#Estimation
#     """
#     n = len(x)
#     variance = x.var()
#     x = x-x.mean()
#     r = correlate(x, x, mode = 'full')[-n:]
#     #assert allclose(r, array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
#     result = r/(variance*(arange(n, 0, -1)))
#     return result
#     #return r

# def pearson(a,b):
# 	corr = cov(a,b)
# 	R = corr[0,1] / ((var(a) * var(b))**0.5)
# 	return R

# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c

# def exponential(p, x):
# 	a,b,c=p
# 	y = a * np.exp(-b * x) + c
# 	return y

# def residuals(p,x,y):
# 	return y - exponential(p,x)

def simulation():

	fire_data = []

	K=float(2)

	N=100	#100

	# this for Rate 10 (reviewer question)
	#f='Avalanche500stepsRate10_K='+str(float(K))+'_E='+str(float(factor_excitation))+'_I='+str(float(factor_inhibition))+'_P='+str(float(factor_neuronexc))+'_N='+str(N)+'_Runs='+str(n_runs)+'.dat'
	#f2='TimeSeries500stepsRate10_K='+str(float(K))+'_E='+str(float(factor_excitation))+'_I='+str(float(factor_inhibition))+'_P='+str(float(factor_neuronexc))+'_N='+str(N)+'_Runs='+str(n_runs)+'.dat'

	f='Bornholdt_Avalanche500steps_K='+str(float(K))+'_N='+str(N)+'.dat'

	f2='Bornholdt_TimeSeries500steps_K='+str(float(K))+'_N='+str(N)+'.dat'


	avalanche_vector=[]
	ACT=[]

	neuron_v=zeros((N))
	neuron_v_new=zeros((N))
	neuron_history=zeros((N))
	weights=np.random.uniform(0,1, size=(N,N))
	fill_diagonal(weights, 0)

	ind=np.random.choice(range(N), int(N*0.2), replace=False)
	weights[:,ind]=weights[:,ind]*-1

	sum_weights=sum(weights)
	weights=weights/sum_weights*N*K


	# w, v = LA.eig(weights)
	# print(max(abs(w)))
	#plot(1,1)
	#show()

	

	Lamb_dyn=[]
	for born in range(10):
		neuron_history=zeros((N))

		for run in range(10):
			### start avalanche
			neuron_v=zeros((N))
			neuron_v[random.randint(0,N-1)]=1 	# background activity original, without Rate in filename
			t=0
			while sum(neuron_v)>0 and t<50:
				
				for i in range(N):
					
					prob_spiking=dot(weights[i,:], neuron_v)
					### Shew 2015
					#prob_spiking=0
					#for j in range(N):
					#	if i!=j:
					#		prob_spiking=prob_spiking + (weights[i,j]*neuron_v[j])
					if prob_spiking>=1.0:
						neuron_v_new[i]=1
						neuron_history[i]=1
					if prob_spiking>random.uniform(0,1) and prob_spiking<1:
						neuron_v_new[i]=1
						neuron_history[i]=1

				# ACT.append(sum(neuron_v))
				fire_data.append(neuron_v_new.copy())
				
				neuron_v[:]=neuron_v_new[:]
				neuron_v_new[:]=0
				#neuron_v[random.randint(0,N-1)]=1 	# background activity original, without Rate in filename
				t=t+1
		#plt.plot(ACT)
		#plt.show()
			
		for jj in range(20):
			rand_neuron=random.randint(0,N-1)
			rand_incoming=random.randint(0,N-1)
			while rand_neuron==rand_incoming:
				rand_incoming=random.randint(0,N-1)
			if neuron_history[rand_neuron]==0:
				#if weights[rand_neuron,rand_incoming]<=0.9:
					weights[rand_neuron,rand_incoming]=np.random.uniform(0,1)#weights[rand_neuron,rand_incoming]+0.1#
			if neuron_history[rand_neuron]==1:
					weights[rand_neuron,rand_incoming]=0#weights[rand_neuron,rand_incoming]-0.1#
					if weights[rand_neuron,rand_incoming]<0:
						weights[rand_neuron,rand_incoming]==0

		# w, v = LA.eig(weights)
		# Lamb_dyn.append(max(abs(w)))

	# plt.plot(Lamb_dyn)
	# plt.show()

	fire_data = torch.tensor(fire_data)


	return fire_data


	#savetxt(f2, ACT)

	#events=histogram(avalanche_vector, (max(avalanche_vector)), density=True)
	#events=events[0]
	#loglog(arange(1,max(avalanche_vector)+1), events, color='red', label='high AED')
	#loglog(arange(1,max(avalanche_vector)+1), arange(1,max(avalanche_vector)+1)**-1.5, color='grey', label='high AED')
	#show()
	#savetxt(f, avalanche_vector)

