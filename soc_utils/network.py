# -*- coding: utf-8 -*-
from __future__ import division
#import matplotlib
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
# from numpy import linalg as LA
import torch
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




def fire(N, K):

    fire_data = []
    Lamb_dyn = []
    connectivity = []

    neuron_v=torch.zeros(N, device=device)
    neuron_v_new=torch.zeros(N, device=device)
    neuron_history=torch.zeros(N, device=device)
    weights=torch.rand((N,N), device=device)
    weights.fill_diagonal_(0)

    ind=np.random.choice(range(N), int(N*0.2), replace=False)
    weights[:,ind]=weights[:,ind]*-1

    sum_weights=torch.sum(weights)
    # not multiplying N
    # weights=weights/sum_weights*K
    weights=weights/sum_weights*N*K

    initial_c = weights.mean().cpu()

    
    # weight update loop
    for born in tqdm(range(30)):
        neuron_history=torch.zeros(N, device=device)

        # run loop with random activities
        for run in range(10):
            ### start avalanche
            neuron_v=torch.zeros(N, device=device)
            idx = torch.randint(0, N, (1,), device=device).item()
            neuron_v[idx]=1 	# background activity original, without Rate in filename
            t=0

            # coverge loop
            while neuron_v.sum()>0 and t<50:
                prob_spiking=torch.matmul(weights, neuron_v)
                rand_vals = torch.rand_like(prob_spiking)

                spike_mask = (prob_spiking >= 1) | (
                    (prob_spiking > 0) & 
                    (prob_spiking < 1) &
                    (rand_vals < prob_spiking)
                )

                neuron_v_new[spike_mask] = 1
                neuron_history[spike_mask] = 1

                ### Shew 2015
                #prob_spiking=0
                #for j in range(N):
                #	if i!=j:
                #		prob_spiking=prob_spiking + (weights[i,j]*neuron_v[j])

            
                fire_data.append(neuron_v_new.detach().cpu().clone())
                
                neuron_v[:]=neuron_v_new[:]
                neuron_v_new[:]=0
                #neuron_v[random.randint(0,N-1)]=1 	# background activity original, without Rate in filename
                t=t+1
            fire_data.append(torch.full((100,), 2))

            
        for _ in range(20):
            rand_neuron=torch.randint(0, N, (1,), device=device).item()
            rand_incoming=torch.randint(0, N, (1,), device=device).item()
            while rand_neuron==rand_incoming:
                rand_incoming=torch.randint(0, N, (1,), device=device).item()
            if neuron_history[rand_neuron]==0:
                #if weights[rand_neuron,rand_incoming]<=0.9:
                    weights[rand_neuron,rand_incoming]=torch.rand((), device=device)#weights[rand_neuron,rand_incoming]+0.1#
            if neuron_history[rand_neuron]==1:
                    weights[rand_neuron,rand_incoming]=0#weights[rand_neuron,rand_incoming]-0.1#
                    if weights[rand_neuron,rand_incoming]<0:
                        weights[rand_neuron,rand_incoming]=0
        
        if born % 200 == 0:
            w = torch.linalg.eigvals(weights)
            Lamb_dyn.append(w.abs().max().item())
        connectivity.append(weights.mean().cpu())


    dict_all = {
        'lamb': Lamb_dyn,
        'connectivity': connectivity,
        'fire_data': fire_data,
        'initial_connectivity': initial_c
    }


    return dict_all




def fire_static(N, K):

    fire_data = []
    Lamb_dyn = []
    connectivity = []

    neuron_v=torch.zeros(N, device=device)
    neuron_v_new=torch.zeros(N, device=device)
    neuron_history=torch.zeros(N, device=device)
    weights=torch.rand((N,N), device=device)
    weights.fill_diagonal_(0)

    # add inhibition
    ind=np.random.choice(range(N), int(N*0.5), replace=False)
    weights[:,ind]=weights[:,ind]*-1

    sum_weights=torch.sum(weights)
    # not multiplying N
    # weights=weights/sum_weights*K
    weights=weights/sum_weights*N*K
    print(torch.sum(weights))

    initial_c = weights.mean().cpu()

    

    # run loop with random activities
    for run in range(1):
        ### start avalanche
        neuron_v=torch.zeros(N, device=device)
        idx = torch.randint(0, N, (1,), device=device).item()
        neuron_v[idx]=1 	# background activity original, without Rate in filename
        t=0

        # coverge loop
        # while neuron_v.sum()>0 and t<1000:
        while t<1000:
            prob_spiking=torch.matmul(weights, neuron_v)
            rand_vals = torch.rand_like(prob_spiking)

            spike_mask = (prob_spiking >= 1) | (
                (prob_spiking > 0) & 
                (prob_spiking < 1) &
                (rand_vals < prob_spiking)
            )

            neuron_v_new[spike_mask] = 1
            neuron_history[spike_mask] = 1

            ### Shew 2015
            #prob_spiking=0
            #for j in range(N):
            #	if i!=j:
            #		prob_spiking=prob_spiking + (weights[i,j]*neuron_v[j])

        
            fire_data.append(neuron_v_new.detach().cpu().clone())
            
            neuron_v[:]=neuron_v_new[:]
            neuron_v_new[:]=0
            #neuron_v[random.randint(0,N-1)]=1 	# background activity original, without Rate in filename
            t=t+1

        
        # if born % 200 == 0:
        #     w = torch.linalg.eigvals(weights)
        #     Lamb_dyn.append(w.abs().max().item())

        connectivity.append(weights.mean().cpu())


    dict_all = {
        'lamb': Lamb_dyn,
        'connectivity': connectivity,
        'fire_data': fire_data,
        'initial_connectivity': initial_c
    }

    return dict_all

    # fire_data = dict_all["fire_data"]
    # fire_sum = np.sum(fire_data, axis=1)

    # return 200-fire_sum[-1]