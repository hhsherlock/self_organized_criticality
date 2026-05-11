import torch, torchvision, sys, argparse, os, shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # only used to save model metrics
import copy
import seaborn as sb
import h5py
import pickle

from time import time, ctime
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from numba.typed import List
from numba import njit

# get neural network and utilities
import NN_set_up
import NN_rloss
import NN_measures
# from PLOT_dynrange_digitrec import forward_stat
import scipy.interpolate as ip
import glob
import re

# get parameters
from PARAM_binary_neuron_network import *

# get paths for saving data
from PATHS import *

#m = cmin
S = Smin_ARG
N = np.sum(n_l)

start_time = time()
print(f"\nStarting {sys.argv[0]} at {ctime()}")

images_to_calculate_slope = 1000

# set data and output paths

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--localdir", dest="local_dir", action="store_true", help="Local directory tree")
parser.add_argument("-n", "--nocomp", dest="no_comp", action="store_true", help="No computations, just plotting already computed data")
parser.add_argument("-c", "--cluster", dest ="cluster_iterations", action="store_true", help="Distribute iterations on cluster")
parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="Loads NN_test.py and checks rewiring")
#parser.add_argument("-cifar10", "--cifar10", dest="cifar10", action="store_true", help="Loads the CIFAR10 dataset")
parser.add_argument("learning_rate", type=float, nargs=1)
parser.add_argument("rewiring_factor", type=float, nargs=1)
parser.add_argument("activity_threshold", type=float, nargs=1)
parser.add_argument("global_run", type=int, nargs=1)
#parser.add_argument("input", nargs=2)
pars_arg = parser.parse_args()

activity_threshold = pars_arg.activity_threshold[0]

print(f'\nUsing rewiring factor {pars_arg.rewiring_factor[0]} and learning rate {pars_arg.learning_rate[0]}\n')

# get paths of training/testing data
if pars_arg.local_dir:
	mnist_path = mnist_path_local
	cifar10_path = cifar10_path_local
else:
	mnist_path = mnist_path_cluster
	cifar10_path = cifar10_path_cluster
	parent_dir = parent_dir_cluster

output_path = mnistpath

if dataset == 'cifar10':
	output_path = output_path + 'cifar10'
# make directory if not existent
if not os.path.exists(output_path):
	# only if not in computation mode
	if not pars_arg.no_comp:
		os.makedirs(output_path)

# derive additional parameters

N = sum(n_l) # number of nodes
l = len(n_l) # number of layers

# print all parameters
asciiart = '''
  ____       _      ____       _      __  __ 
 |  _ \     / \    |  _ \     / \    |  \/  |  _
 | |_) |   / _ \   | |_) |   / _ \   | |\/| | |_|
 |  __/   / ___ \  |  _ <   / ___ \  | |  | |  _
 |_|     /_/   \_\ |_| \_\ /_/   \_\ |_|  |_| |_|
'''
print(asciiart)
with open('PARAM_binary_neuron_network.py', 'r') as file:
	for line in file:
		print(line, end='')
print('\n\n\n')

# get device (cpu/gpu)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # check available device
print('torch.device = {}'.format(device))

# set up iteration variables

if not pars_arg.no_comp:
	if pars_arg.cluster_iterations:
		# parse global_run variable from cluster
		global_run = pars_arg.global_run[0] # parsed as list
		
		# only computer one iteration on this machine
		iterations = 1

def cat_lists_iterate(l):
	empty_list = []
	for elements1 in l:
		for elements2 in elements1:
			empty_list.append(elements2)

	return empty_list

# transform data to tensor and map to [-1,1]		

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))]) # normalize with 0.5 mean and 0.5 stdev , torchvision.transforms.Normalize((0.5), (0.5))

# download datasets

if dataset=='cifar10':
	trainset = torchvision.datasets.CIFAR10(cifar10_path, download=True, train=True, transform=transform)
	valset = torchvision.datasets.CIFAR10(cifar10_path, download=True, train=False, transform=transform)
else:
	trainset = torchvision.datasets.MNIST(mnist_path, download=True, train=True, transform=transform)
	valset = torchvision.datasets.MNIST(mnist_path, download=True, train=False, transform=transform)

# dont use all the data (for debugging) 

#trainset = torch.utils.data.Subset(trainset, torch.arange(6000))
#valset = torch.utils.data.Subset(valset, torch.arange(1000))

# initialize useful arrays and variables

# training metrics
training_loss = torch.zeros(epochs, iterations)
training_accuracy = torch.zeros(epochs, iterations)
training_auc_micro = torch.zeros(epochs, iterations)
training_auc_classes = torch.zeros((epochs, n_l[-1], iterations))

# test metrics
validation_accuracy = torch.zeros(epochs, iterations)
validation_auc_micro = torch.zeros(epochs, iterations)
validation_auc_classes = torch.zeros((epochs, n_l[-1], iterations))
test_loss_save = torch.zeros(epochs, iterations)

# averaged weights per epoch
average_weights_epoch = torch.zeros((epochs, iterations))
stdev_weights_epoch = torch.zeros((epochs, iterations))

# average activations per epoch
average_act_epoch = torch.zeros((epochs, iterations))
stdev_act_epoch = torch.zeros((epochs, iterations))

# preallocate list for networks
net_save = [0]*iterations

# save network output
output_save = []

# how many classes does the dataset have?
n_classes = n_l[-1]

def flatten_arbitrary_depth(lst):
	for item in lst:
		if isinstance(item, (list, tuple)):
			yield from flatten_arbitrary_depth(item)
		elif isinstance(item, np.ndarray):
			yield from item.ravel()
		else:
			yield item

####################################################
## diese verkabelungsregel ist ein erster entwurf ##
## multiplikativ ###################################
## und nicht optimiert mit numba ###################
## aukommentiert am 27.9.23 ########################
####################################################

# from typing import List
# # @torch.jit.script
# def bornholdt_rewiring(activations: List[torch.Tensor], weights: List[torch.Tensor], fraction_rewired: float, activity_threshold: float, device: torch.device):
#	  
#	  with torch.no_grad():
#		  
#		  weights_new = []
#		  
#		  for layer, act_batch in enumerate(activations): # iterate layers
#			  
#			  if layer==0 or layer==len(activations)-1: continue # skip first/last layer
#			  
#			  absolute_rewired_layer = int(len(act_batch[0,:]) * fraction_rewired) # how many neurons in each layer? 
#			  
#			  # choose neurons and links 
#			  rand_neurons = torch.randperm(len(act_batch[0,:]))[:absolute_rewired_layer].to(device) # without replacement
#			  rand_incoming = torch.randint(0, weights[layer-1].shape[1], (absolute_rewired_layer,)).to(device) # with replacement
#			  
#			  # this is where the weight updates for each samples are saved
#			  layerweights_new = weights[layer-1].unsqueeze(2).repeat(1,1,act_batch.shape[0])
#			  
#			  for nsample, act_sample in enumerate(act_batch): # iterate samples
#				  
#				  # calculate weight updates each sample/layer
#				  active_neurons = torch.abs(act_sample[rand_neurons]) > activity_threshold
#				  
#				  # save weight updates 
#				  layerweights_new[rand_neurons[active_neurons], rand_incoming[active_neurons], nsample] = weights[layer-1][rand_neurons[active_neurons], rand_incoming[active_neurons]] * 0.5
#				  layerweights_new[rand_neurons[~active_neurons], rand_incoming[~active_neurons], nsample] = weights[layer-1][rand_neurons[~active_neurons], rand_incoming[~active_neurons]] * 2.0
#			  
#			  # transfer weight updates to weight matrix 
#			  weights[layer-1] = layerweights_new.mean(axis=2) # average weight updates
#							  
#	  return weights

#######################################################
## diese verkablungsregel ist multiplikativ, und ######
## wurde bis 27.9.23 in allen simulationen verwendet ##
## mit numba optimiert ################################
## auskommentiert am 27.9.23 ##########################
#######################################################

# @njit
# def bornholdt_rewiring_numpy(activations, weights, fraction_rewired, activity_threshold, rconst):
#	  
#	  # neuron_history = [act[...,None].abs()>(1.7159/2) for layer, act in enumerate(activations)]
#	  
#	  for layer, act_batch in enumerate(activations): # iterate layers
#		  
#		  if layer==0 or layer==len(activations)-1: continue # skip first/last layer
#		  
#		  absolute_rewired_layer = int(len(act_batch[0,:]) * fraction_rewired) # how many neurons in each layer?
#		  
#		  # choose neurons and links
#		  rand_neurons = np.random.choice(np.arange(len(act_batch[0,:])), absolute_rewired_layer, replace=False)
#		  rand_incoming = np.random.choice(np.arange(weights[layer-1].shape[1]), absolute_rewired_layer, replace=True)
#		  
#		  newweights = np.empty((absolute_rewired_layer, len(act_batch)))
#		  
#		  for nsample, act_sample in enumerate(act_batch): # iterate samples
#			  # calculate weight updates each sample/layer
#			  active_neurons = np.abs(act_sample[rand_neurons]) > activity_threshold
#			  
#			  for nneuron, (inneuron, outneuron, active) in enumerate(zip(rand_neurons, rand_incoming, active_neurons)):
#				  if active:
#					  newweights[nneuron, nsample] = weights[layer-1][inneuron, outneuron] * 1/rconst
#				  else:
#					  newweights[nneuron, nsample] = weights[layer-1][inneuron, outneuron] * rconst
#			  
#		  # average weights over batch
#		  newweights = newweights.sum(axis=1)/len(act_batch)
#		  for nneuron, (inneuron, outneuron) in enumerate(zip(rand_neurons, rand_incoming)):
#			  weights[layer-1][inneuron, outneuron] = newweights[nneuron]
#		  
#	  return weights

#######################################################
## diese verkablungsregel ist ein neuer versuch alle ##
## neuronen korrekt zu verkabeln, additiv, und ########
## bezieht die präsynaptischen neuronen mit ein ######
## verwendet ab dem 27.9.23 ###########################
#######################################################

# # @njit
# def bornholdt_rewiring_numpy(activations, weights, fraction_rewired, activity_threshold, rconst):
#	  
#	  # neuron_history = [act[...,None].abs()>(1.7159/2) for layer, act in enumerate(activations)]
#	  
#	  for layer, act_batch in enumerate(activations): # iterate layers
#		  
#		  if layer==0 or layer==len(activations)-1: continue # skip first/last layer
#		  
#		  absolute_rewired_layer = int(len(act_batch[0,:]) * fraction_rewired) # how many neurons in each layer?
#		  
#		  # choose neurons and links
#		  rand_neurons = np.random.choice(np.arange(len(act_batch[0,:])), absolute_rewired_layer, replace=False)
#		  rand_incoming = np.random.choice(np.arange(weights[layer-1].shape[1]), absolute_rewired_layer, replace=True)
#		  print('layer:', layer, 'rand_neurons = ', rand_neurons, f'({act_batch[0][rand_neurons]})', 'rand_incoming = ', rand_incoming, f'({activations[layer-1][0][rand_incoming]})')
#		  
#		  newweights = np.empty((absolute_rewired_layer, len(act_batch)))
#		  
#		  for nsample, act_post_sample in enumerate(act_batch): # iterate samples
#			  
#			  # increase or decrease activation of postsynaptic neuron?
#			  condition1 = act_post_sample[rand_neurons] > activity_threshold # activity>thresh
#			  condition2 = (-activity_threshold < act_post_sample[rand_neurons]) & (act_post_sample[rand_neurons] <= 0) # -thresh<activity<=0
#			  decrease_postsyn_neuron = condition1 | condition2 # either condition1 or condition2
#			  
#			  # is the presynaptic neuron negative or positive?
#			  act_pre_sample = activations[layer-1][nsample][rand_incoming] # fetch activation of presynaptic neuron
#			  presyn_is_positive = act_pre_sample > 0
#			  
#			  # iterate rewired neurons
#			  for nneuron, (inneuron, outneuron, decrease_post, positive_pre) in enumerate(zip(rand_neurons, rand_incoming, decrease_postsyn_neuron, presyn_is_positive)):
#				  
#				  # if postsynaptic neurons shall be less active
#				  if decrease_post:
#					  if act_pre_sample[nneuron] > 0:
#						  newweights[nneuron, nsample] = weights[layer-1][inneuron, outneuron] - rconst  
#					  if act_pre_sample[nneuron] < 0:
#						  newweights[nneuron, nsample] = weights[layer-1][inneuron, outneuron] + rconst 
#				  
#				  # if postsynaptic neurons shall be more active 
#				  else:
#					  if act_pre_sample[nneuron] > 0:
#						  newweights[nneuron, nsample] = weights[layer-1][inneuron, outneuron] + rconst 
#					  if act_pre_sample[nneuron] < 0:
#						  newweights[nneuron, nsample] = weights[layer-1][inneuron, outneuron] - rconst 
#				  print('weights old:', weights[layer-1][inneuron, outneuron])
#				  print('weights new:', newweights[nneuron, nsample])
#				  # # plus or minus the additive constant c?
#				  # updaterule = +1 if (decrease_post == positive_pre) else -1
#				  # 
#				  # # apply the rule and update the weights
#				  # newweights[nneuron, nsample] = weights[layer-1][inneuron, outneuron] + rconst*updaterule
#				  
#				  # clip weights if signs changes during rewiring
#				  if np.sign(newweights[nneuron, nsample]) != np.sign(weights[layer-1][inneuron, outneuron]) and newweights[nneuron, nsample]!=0 and weights[layer-1][inneuron, outneuron]!=0:
#					  newweights[nneuron, nsample] = 0
#				   
#		  # average weights over batch
#		  newweights = newweights.sum(axis=1)/len(act_batch)
#		  for nneuron, (inneuron, outneuron) in enumerate(zip(rand_neurons, rand_incoming)):
#			  weights[layer-1][inneuron, outneuron] = newweights[nneuron]
#		  
#	  return weights

######################################
## This rewiring function defines   ##
## neuron activity by changes       ##
## over time, used since 9 oct 2023 ##
######################################

# @njit
# def bornholdt_rewiring_numpy(activations, weights, fraction_rewired, fraction_links, activity_threshold, rconst):
# 	
# 	# neuron_history = [act[...,None].abs()>(1.7159/2) for layer, act in enumerate(activations)]
# 	
# 	for layer, act_batch in enumerate(activations): # iterate layers
# 		
# 		if layer==0 or layer==len(activations)-1: continue # skip first/last layer
# 		
# 		absolute_rewired_layer = int(len(act_batch[0,:]) * fraction_rewired) # how many neurons in each layer?
# 		absolute_links_layer = int(weights[layer-1].shape[1] * fraction_links) # how many incoming links are changed? 
# 		
# 		# choose neurons and links
# 		rand_postsyn = np.random.choice(np.arange(len(act_batch[0,:])), absolute_rewired_layer, replace=False)
# 		# rand_presyn = np.random.choice(np.arange(weights[layer-1].shape[1]), absolute_rewired_layer, replace=True)
# 		# rand_presyn = np.random.choice(
# 		# 	np.arange(weights[layer-1].shape[1]),
# 		# 	(absolute_rewired_layer, absolute_links_layer),
# 		# 	replace=False)
# 		rand_presyn = np.zeros((absolute_rewired_layer, absolute_links_layer), dtype=np.int32)
# 		for i in rand_presyn:
# 			i[:] = np.random.choice(np.arange(weights[layer-1].shape[1]), absolute_links_layer, replace=False)
# 		
# 		# rewire
# 		# for presyn, postsyn in zip(rand_presyn, rand_postsyn):
# 		for presyn, postsyn in zip(rand_presyn.flatten(), rand_postsyn.repeat(absolute_links_layer)):
# 			
# 			# calc activity from standard deviation
# 			active = act_batch[:,postsyn].std() > activity_threshold
# 			
# 			# change weight
# 			current_weight = np.array([weights[layer-1][postsyn,presyn]])
# 			
# 			if active: # active neuron
# 				
# 				# only chose non-zero link if neuron is active
# 				while current_weight == 0:
# 					presyn = np.random.choice(np.arange(weights[layer-1].shape[1]), 1)[0]
# 					current_weight = np.array([weights[layer-1][postsyn,presyn]])
# 				
# 				weights[layer-1][postsyn,presyn] = ((np.abs(current_weight)-rconst).clip(0,None) * np.sign(current_weight))[0]
# 			
# 			else: # inactive neuron
# 				
# 				if current_weight != 0:
# 					weights[layer-1][postsyn,presyn] = ((np.abs(current_weight)+rconst) * np.sign(current_weight))[0]
# 				if current_weight == 0:
# 					weights[layer-1][postsyn,presyn] = (rconst * np.random.choice(np.array([-1,1])))
# 		
# 	return weights

# check if no_comp flag is given
if not pars_arg.no_comp:

	for n_run in range(iterations):
		
		# plot the loss as function of the weights
		# figloss, axloss = plt.subplots()
		
		# set global_run variable for saving
		if not pars_arg.cluster_iterations:
			global_run = n_run

		# where to calculate rloss
		calcrloss = torch.arange(start=0, end=epochs, step=rloss_epochs)
		#gaussian_mean = []
		rloss = []
		lyapunov = []
		all_lyapunovs = []

		# pre-allocate array to save effective rewiring constant, scaled with rloss
		rconst_eff = []

		# pre-allocate array to save weightdiff befor/after SGD and SOC weightupdate
		weightdiff_SGD_save = []
		weightdiff_SOC_save = []
		gradients_save = []
		loss_save = []
		
		# load datasets
		
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
		valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=True)
		
		dataiter = iter(trainloader)
		images, labels = next(dataiter)
		
		# load images without batching for rloss calc
		trainloader_nobatch = torch.utils.data.DataLoader(trainset, batch_size=None, shuffle=True)
		# images_nobatch, _ = next(iter(trainloader_fullbatch))
	   
		
		print('\nimages.shape = {}'.format(images.shape))
		try:
			print('labels.shape = {}'.format(labels.shape))
		except AttributeError:
			print('type(labels) = {}'.format(type(labels)))

		print('number of classes = {}'.format(n_classes))

		# how many batches in training examples?
		nr_batches = sum(1 for batch in trainloader)
		print('number of batches = {}'.format(nr_batches))

		# display grid of images
		
		#figure = plt.figure()
		#num_of_images = 60
		#for index in range(1, num_of_images + 1):
		#	plt.subplot(6, 10, index)
		#	plt.axis('off')
		#	plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
		#figure.savefig(f"{output_path}rand_mnist_img.png")
			
		n_total_steps = len(trainloader)
		n_total_steps_test = len(valloader)
		precision = 6 # precision of printed accuracy / loss

		if model=='shew09':
			raise TypeError('Model Shew09 not implemented as torch network')
		else:
			# net = NN_set_up.net_init(n_l, model, activity_threshold, rfmin, batchnorm, bn_fixed, gamma, beta).to(device) # instantiating network
			net = NN_set_up.net_init(
				n_l,
				model,
				activity_threshold,
				rfmin,
				batchnorm,
				bn_fixed,
				gamma,
				beta,
				input_transform,
				it_fixed,
				scale,
				shift).to(device) # instantiating network
		
		if n_run == 0: print('\nNN Model:\n{}\n'.format(net))
		
		# weight initialization
		
		net = NN_set_up.uniform_weights_to_tensor(net, add_shift=addshift, mul_shift=mulshift) # set uniformly distributed weights
		# net = NN_set_up.uniform_weights_to_tensor(net, low_bound=-0.153, high_bound=0.153, add_shift=addshift, mul_shift=1)
		
		# if critical_init:
		# 	for l, layer in enumerate(net.linears):
		# 		if l == 0 or l == len(net.linears)-1: continue
		# 		layer.weight /= torch.linalg.eigvals(layer.weight).abs().max()
		
		# for layer in net.linears:
		#	  print(f'layer min weight {layer.weight.min()}')
		#	  print(f'layer max weight {layer.weight.max()}')
			
		# make directory for each run
		
		if not os.path.exists(output_path + f'traces/run_{global_run}/'):
			os.makedirs(output_path + f'traces/run_{global_run}/')
		
		torch.save(
			torch.nn.utils.parameters_to_vector(net.parameters()),
			output_path + f'traces/run_{global_run}/initial_parameters.pt')
		
		# define criterion and optimizer
		
		criterion = torch.nn.NLLLoss()
		optimizer = torch.optim.SGD(net.parameters(), lr=pars_arg.learning_rate[0]) # momentum = 0.5
		
		# define multiplicative learning rate
		
		mult_func = lambda epoch: 0.997
		lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=mult_func)

		# save model to file, untouched model (no SGD, no rewiring)
		net_filename = f'my_mnist_model.iter_{global_run}.epoch_0.pt'
		torch.save(net, output_path+net_filename)
		
		net(images.to(device).view(images.shape[0], -1))
		
		# plot the activity (stdev)
		fig, ax = plt.subplots()
		for layer, batchneurons in enumerate(net.trace[:-1]):
			ax.scatter([layer]*batchneurons.shape[1], batchneurons.cpu().std(axis=0), alpha=0.01, c='black', s=100)
		ax.set(xlabel='layer', ylabel='activity (stdev)', title='MNIST input')
		fig.savefig(output_path+f'activityMNIST.iter{global_run}.epoch0.png')
		plt.close()
		
		if local_rewiring:
			
			borninput = (torch.rand(batchsize, n_l[0], device=device)*2-1)*1e-9
			net(borninput.to(device))
			bornact = [trace.detach().cpu().numpy() for trace in net.trace]
			
			# plot the activity (stdev)
			fig, ax = plt.subplots()
			# for layer, batchneurons in enumerate(bornnet.trace[:-1]):
			for layer, batchneurons in enumerate(bornact[:-1]):
				ax.scatter([layer]*batchneurons.shape[1], batchneurons.std(axis=0), alpha=0.01, c='black', s=100)
			ax.set(xlabel='layer', ylabel='activity (stdev)', title='Bornholdt input')
			fig.savefig(output_path+f'activityBORN.iter{global_run}.epoch0.png')
			plt.close()
		
		# save slope of c/DR fit
		# aline = torch.zeros(epochs)
		aline_lle = torch.zeros(epochs, iter_losstest)
		
		# rloss_vec = torch.full((epochs+rloss_windowsize,), float('nan'))
		sigma_vec = []
		
		#time0 = time()
		for e in range(epochs):
			
			# training loss
			running_loss = 0.
			running_correct = 0
			running_all = 0

			# test loss
			test_running_loss = 0.
			#test_running_correct = 0
			#test_running_all = 0
			
			y_true = [] # used for AUC calculation
			global y_score
			y_score = [] # used for AUC calculation
			
			# set string where data is saved
			folder_string = f'traces/run_{global_run}/epoch_{e}/'
			
			# create folder for data
			if not os.path.exists(output_path + folder_string):
				os.makedirs(output_path + folder_string)
			
			# pre-allocate arrays for rewiring data
			rewiring_activity = []
			rewiring_oldweights = np.zeros((nr_batches, nr_neurons))
			rewiring_newweights = np.zeros((nr_batches, nr_neurons))
			rewiring_preindices = np.zeros((nr_batches, nr_neurons))
			rewiring_postindices = np.zeros((nr_batches, nr_neurons))
			
			for train_step, (images, labels) in enumerate(trainloader):
				
				# set model to training mode
				net.train()
				
				# Flatten MNIST images into a 784 long vector
				images = images.view(images.shape[0], -1)
				
				if spontaneous_activity:
					images = images.uniform_(-1, 1) * 1e-9
		
				# port images / labels to device
				images = images.to(device)
				labels = labels.to(device)
				
				# Training pass
				optimizer.zero_grad()
				
				output = net(images)
				
				loss = criterion(output, labels)
				loss_save.append(loss.item())

				weights_prelearning = [copy.deepcopy(layer.weight.detach().cpu().numpy()) for layer in net.linears]
				# weights_prelearning = [layer.weight.detach() for layer in net.linears]
				# weights_prelearning = [layer.weight.detach() for layer in net.linears]
				# weights_prelearning = [copy.deepcopy(layer.weight.detach()) for layer in net.linears]
				# torch.save(weights_prelearning, f"/home/vocks/projects/binary_neuron_network/check/weightspre_gpu.epoch{e}.trainstep{train_step}.pt")
				
				if learning:
					# This is where the model learns by backpropagating
					loss.backward()
					
					# And optimizes its weights here
					optimizer.step()
				
				# calculate metrics
		
				running_loss += loss.item()
				
				_, predicted = torch.max(output.data, 1)
				running_correct += (predicted == labels).sum().item()
				running_all += len(labels)
				
				y_true.append(label_binarize(labels.cpu(), classes=[i for i in range(n_classes)]))
				y_score.append(output.detach())
				
				# print the loss gradients wrt. weights
				if learning==True:
					gradients = [layer.weight.grad.detach().cpu().numpy().flatten() for layer in net.linears]
					gradients_save.append(np.mean(np.abs(np.concatenate(gradients))))
				
				# get activations
				activations = [trace.detach().cpu().numpy() for trace in net.trace]
				# activations = [copy.deepcopy(trace.detach()) for trace in net.trace]
				
				# get weights
				weights = [copy.deepcopy(layer.weight.detach().cpu().numpy()) for layer in net.linears]
				# weights = [layer.weight.detach() for layer in net.linears]
				# torch.save(weights, f"/home/vocks/projects/binary_neuron_network/check/weightspost_gpu.epoch{e}.trainstep{train_step}.pt")
				# weights_save.append(np.mean(np.abs(np.concatenate(weights))))
				
				# weights = [copy.deepcopy(layer.weight.detach()) for layer in net.linears]
				
				# plot loss as function of the weights
				# flatweights = [copy.deepcopy(layer.weight.detach().cpu().numpy().flatten()) for layer in net.linears]
				# flatweights = [w.flatten() for w in weights]
				# axloss.scatter(np.concatenate(flatweights).mean(), [loss.item()])
				# axloss.scatter(torch.concat(flatweights).mean(), [loss.item()])

				# weight difference SGD
				weightdiff_SGD = [(w_post - w_pre).flatten() for w_post, w_pre in zip(weights, weights_prelearning)] # get difference
				weightdiff_SGD = np.mean(np.abs(np.concatenate(weightdiff_SGD))) # absolute values, mean
				# weightdiff_SGD = torch.mean(torch.abs(torch.concat(weightdiff_SGD)))
				weightdiff_SGD_save.append(weightdiff_SGD)
				# print('weightdiff_SGD = ', weightdiff_SGD)
				# weightdiff_SGD_save.append(weightdiff_SGD.cpu())




				# rloss
				if train_step==0:
					if scale_rconst:
						
						# scale rconst using rolling averga of sigma
						# sigmanet = copy.deepcopy(net)
						sigmainput = (torch.rand(batchsize, n_l[0], device=device)*2-1)*1e-9
						net(sigmainput)
						# net(images)
						layer_activity = torch.tensor([layer.std(axis=0).mean() for layer in net.trace])
						sigmadyn_layer = ((layer_activity[1:])/(layer_activity[:-1]))
						sigmadyn = sigmadyn_layer[1:-1].mean()
						sigma_vec.append(sigmadyn)
						# print('rloss_vec =', rloss_vec)
						# print('len(rloss_vec) =', len(rloss_vec))
						
						# if len(sigma_vec) < rloss_windowsize:
						# 	rloss.append(torch.tensor(float('nan')))
						# else:
						# 	# running_trainiter = e*len(trainloader) + train_step
						# 	# rloss_window = range(running_trainiter-rloss_windowsize+1, running_trainiter+1)
						# 	rloss_window = range(e-rloss_windowsize+1, e+1)
						# 	
						# 	# rloss = (1-torch.tensor([rloss_vec[rloss_window]]).mean()).abs()
						# 	window_sigma = torch.tensor(sigma_vec)[rloss_window].mean()
						# 	#rloss = torch.exp(-window_sigma)-0.5
						# 	rloss_val = (window_sigma-1).abs().clamp(min=0, max=1)
						# 	
						# 	if rloss_val < rloss_thresh: rloss_val = torch.tensor(0) # threshold
						# 	rloss.append(rloss_val)
						# 	
						# 	print('sigma_vec =', sigma_vec)
						# 	print('rloss =', rloss)
						
						rloss_val = (sigmadyn-1).abs().clamp(min=0, max=1)
						if rloss_val < rloss_thresh:
							rloss_val = torch.tensor(0) # threshold
							# local_rewiring = False # oncle critical stop rewiring
						rloss.append(rloss_val)
						
						rconst_eff_val = torch.tensor(pars_arg.rewiring_factor[0]) if rloss[-1].isnan() else pars_arg.rewiring_factor[0]*rloss[-1] * rfscale
						
					else:
						# rconst_eff_val = pars_arg.rewiring_factor[0] # constant
						rconst_eff_val = torch.tensor(pars_arg.rewiring_factor[0]) # constant
					
					rconst_eff.append(rconst_eff_val) # save
				# print('rconst_eff =', rconst_eff)
				# print('rloss =', rloss)
				
				
				
				# print('sigmadyn =', sigmadyn)
				# print('rconst_eff_val =', rconst_eff_val)
				# rconst_eff.append(rconst_eff_val) # save








				# # fit multivariate gaussian distribution =============================================================
				# if e in calcrloss and train_step==0:

				# 	# copy network for lle calc
				# 	llenet = copy.deepcopy(net)
				# 	
				# 	# forward pass
				# 	lle_input_vec = (torch.rand(batchsize, n_l[0], device=device)*2-1)*1e-9
				# 	llenet(lle_input_vec)
				# 	growthrate, lyapunovs, activations_llecalc, jacobi_sample0, _, Rdiag = llenet.lle()
				# 	
				# 	all_lyapunovs.append(lyapunovs)
				# 	max_lyapunov = lyapunovs.max()
				# 	growthrate = growthrate.nanmean(axis=1)

				# 	torch.save(activations_llecalc, output_path + f'llecalc.activations_epoch{e}.pt')
				# 	torch.save(jacobi_sample0, output_path + f'llecalc.jacobi.sampleneuron0_epoch{e}.pt')
				# 	torch.save(Rdiag, output_path + f'llecalc.Rdiag_epoch{e}.pt')
				# 	
				# 	# save growthrate
				# 	torch.save(growthrate, output_path+f'growthrate.iter_{n_run}.epoch_{e}.pt')
				# 	
				# 	lyapunov.append(lyapunovs.nanmean().item())
				# 
				# 	# scale rewiring constant with rloss
				# 	if scale_rconst:
				# 		
				# 		# rc_vals = torch.logspace(0,2,steps=11)
				# 		# # test_dynamicrange = torch.zeros(len(rc_vals), iter_losstest)
				# 		# test_lle = torch.zeros(iter_losstest, len(rc_vals))
				# 		# 
				# 		# lle_nobatch = torch.zeros((images_to_calculate_slope, iter_losstest, len(rc_vals)))
				# 		# 
				# 		# # iterate the whole test 
				# 		# for i_iter in range(iter_losstest):
				# 		# 	
				# 		# 	# iterate different rewiring constants c 
				# 		# 	for i_rc, rc_test in enumerate(rc_vals):
				# 		# 		# print('scale test. iter:', i_iter, '# rc_val', i_rc)
				# 		# 		
				# 		# 		# copy test network
				# 		# 		testnet = copy.deepcopy(net)
				# 		# 		
				# 		# 		fraction_rewired = nr_neurons/np.sum(n_l[1:-1])
				# 		# 		# fraction_links = 0.1
				# 		# 		
				# 		# 		# rewire the network with different constants c
				# 		# 		weights_new = bornholdt_rewiring_numpy(
				# 		# 			list(activations), 
				# 		# 			list(weights), 
				# 		# 			fraction_rewired,
				# 		# 			fraction_links,
				# 		# 			activity_threshold,
				# 		# 			rc_test.item())
				# 		# 		
				# 		# 		# update weights in network instance
				# 		# 		torch.nn.utils.vector_to_parameters(torch.concat([torch.tensor(w.flatten()).float().to(device) for w in weights_new]), testnet.parameters())
				# 		# 		
				# 		# 		# calc LLEs for rewired testnet (all images in one batch)
				# 		# 		# testnet(images) # make forward pass only one batch
				# 		# 		testnet(lle_input_vec)
				# 		# 		# testnet(images_fullbatch) # make forward pass
				# 		# 		
				# 		# 		# test_lle[i_iter, i_rc] = testnet.lle()[1].nanmean() # take mean lyapunov
				# 		# 		get_all_lyapunovs = testnet.lle()[1]
				# 		# 		
				# 		# 		test_lle[i_iter, i_rc] = get_all_lyapunovs[~torch.isnan(get_all_lyapunovs)].mean() # take max lyapunov
				# 		# 
				# 		# # fit line LLEs for each iteration and plot
				# 		# 
				# 		# figtest, axtest = plt.subplots()
				# 		# for iter_tlle, tlle in enumerate(test_lle):
				# 		# 	aline_lle[e, iter_tlle], bline_lle = np.polyfit(rc_vals, tlle, 1)
				# 		# 	axtest.scatter(rc_vals, tlle, s=200)
				# 		# 	axtest.plot(rc_vals, aline_lle[e, iter_tlle]*rc_vals+bline_lle, linewidth=7)
				# 		# 	axtest.set_xlabel('Bornholdt c')
				# 		# 	axtest.set_ylabel('LLE')
				# 		# figtest.savefig(output_path + f'rc_test_lle.iter_{global_run}.epoch_{e}.png', bbox_inches='tight')
				# 		# plt.close()
				# 		# 
				# 		# with open(output_path + f'test_lle.iter{global_run}.epoch{e}.pkl', 'wb') as handle:
				# 		# 	pickle.dump(test_lle, handle, protocol=pickle.HIGHEST_PROTOCOL)
				# 		# 
				# 		# with open(output_path + f'rc_vals.iter{global_run}.epoch{e}.pkl', 'wb') as handle:
				# 		# 	pickle.dump(rc_vals, handle, protocol=pickle.HIGHEST_PROTOCOL)
				# 	
				# 		# # rloss_val = aline_lle[e,:].abs().mean()
				# 		# rloss_val = aline_lle[e,:].mean()
				# 		# # rloss_val_std = aline_lle[e,:].abs().std()
				# 		# rloss_val_std = aline_lle[e,:].std()
				# 		# # threshold_in_std = 1
				# 		# # if rloss_val < 1*aline_lle[e,:].std():
				# 		# if (rloss_val+rloss_threshold_in_std*rloss_val_std > 0) and (rloss_val-rloss_threshold_in_std*rloss_val_std < 0):
				# 		# 	rloss_val = torch.tensor(0.)
				# 		# else:
				# 		# 	rloss_val = torch.tensor(1.)
				# 		# 
				# 		# rloss.extend([rloss_val])
				# 	
				# 		# # rconst_eff_val = pars_arg.rewiring_factor[0]*rloss_val # effective rewiring constant, scaled with rloss
				# 		# # rconst_eff_val = 1.0 + rloss_val
				# 		# # rconst_eff_val = (1 + (pars_arg.rewiring_factor[0]-1)*rloss_val*rfscale).item()
				# 		# # rconst_eff_val = (1 + rloss_val*rfscale).item()
				# 		# # rconst_eff_val = torch.exp(rloss_val).item()*rfscale
				# 		# # rconst_eff_val = 1 + rfscale*rloss_val.item()
				# 		# rconst_eff_val = pars_arg.rewiring_factor[0] if rloss_val==1 else 1
				# 		
				# 		
				# 		
				# 		# scale rconst using rolling averga of sigma
				# 		sigmanet = copy.deepcopy(net)
				# 		sigmainput = (torch.rand(batchsize, n_l[0], device=device)*2-1)*1e-9
				# 		sigmanet(sigmainput)
				# 		layer_activity = torch.tensor([layer.std(axis=0).mean() for layer in sigmanet.trace])
				# 		sigmadyn_layer = ((layer_activity[1:])/(layer_activity[:-1]))
				# 		sigmadyn = sigmadyn_layer[1:-1].mean()
				# 		# rloss_vec[e+rloss_windowsize] = sigmadyn
				# 		rloss_vec.append(sigmadyn)
				# 		print('rloss_vec =', rloss_vec)
				# 		if len(rloss_vec) < rloss_windowsize:
				# 			rloss = torch.tensor([float('nan')])
				# 		else:
				# 			running_trainiter = e*len(trainloader) + train_step
				# 			rloss_window = range(running_trainiter-rloss_windowsize+1, running_trainiter+1)
				# 			rloss = (1-torch.tensor([rloss_vec[rloss_window]]).mean()).abs()
				# 		
				# 		rconst_eff_val = pars_arg.rewiring_factor[0] if rloss.isnan() else pars_arg.rewiring_factor[0]*rloss
				# 		
				# 		
				# 		
				# 			
				# 	else:
				# 		# rconst_eff_val = pars_arg.rewiring_factor[0] # constant
				# 		rconst_eff_val = pars_arg.rewiring_factor[0] # constant
				# 	
				# 	rconst_eff.append(rconst_eff_val) # save
				# 	
				# 	
				# 	
				# 	
				# 	print('sigmadyn =', sigmadyn)
				# 	print('rconst_eff_val =', rconst_eff_val)
				# 	rconst_eff.append(rconst_eff_val) # save
					
				# local rewiring ========================================================================================================
				if local_rewiring and e<stop_rewiring:
					## create folder for data
					#if not os.path.exists(output_path + folder_string):
					#	 os.makedirs(output_path + folder_string)
					
					# get activity threshold
					if activity_threshold=="input":
						activity_threshold_val = float(torch.mean(torch.abs(images)))
					else:
						activity_threshold_val = activity_threshold
					
					# apply bornholdt 2000 rules
					weights_preborn = [copy.deepcopy(layer.weight.detach()) for layer in net.linears]
					
					selected_neurons = torch.full((nr_neurons,batchsize), float('nan'))
					fraction_rewired = nr_neurons/np.sum(n_l[1:-1])
					# fraction_links = 0.1
					
					# weights_new = bornholdt_rewiring(activations, weights, fraction_rewired, activity_threshold, device)
					
					# bornholdt on small inputs 10^-9
					# bornnet = copy.deepcopy(net)
					bornnet = net
					# borninput = (torch.rand((batchsize, n_l[0]), device=device)*2-1)*1e-9
					borninput = images
					# borninput = (torch.rand((100, n_l[0]), device=device)*2-1)*1e-9
					# from time import time
					# time_born0 = time()
					bornnet(borninput)
					# time_born1 = time()
					# print('timeborn1 - timeborn0 =', time_born1-time_born0)
					# print('train_step = ', train_step)
					fraction_neurons = nr_neurons/np.sum(n_l[1:-1])
					net.bstep = rconst_eff_val.to(device)
					net.bornholdt(borninput, fraction_neurons, fraction_links)
					# net.bornholdt(images, fraction_neurons, fraction_links)
					
					bornact = [trace.detach().cpu().numpy() for trace in bornnet.trace]
					# 
					# 
					# weights_new = bornholdt_rewiring_numpy(
					# 	list(bornact), 
					# 	list(weights), 
					# 	fraction_rewired,
					# 	fraction_links,
					# 	activity_threshold,
					# 	rconst_eff_val)
					# 
					# # update weights in network instance
					# torch.nn.utils.vector_to_parameters(torch.concat([torch.tensor(w.flatten()).float().to(device) for w in weights_new]), net.parameters())
							
								
					weights_postborn = [copy.deepcopy(layer.weight.detach()) for layer in net.linears]
					
					# # get weights after SOC rewiring
					# weights_postSOC = [copy.deepcopy(layer.weight.detach().cpu().numpy()) for layer in net.linears]
					
					# weight difference SOC
					# weightdiff_SOC = [(w_post - w_pre).flatten() for w_post, w_pre in zip(weights_postSOC, weights)] # get difference
					weightdiff_SOC = [(w_post - w_pre).flatten().cpu().numpy() for w_pre, w_post in zip(weights_preborn, weights_postborn)]
					
					weightdiff_SOC = np.mean(np.abs(np.concatenate(weightdiff_SOC))) # absolute values, mean
					weightdiff_SOC_save.append(weightdiff_SOC)
					
				# save trace matrix every save_step  time steps ================================================================
				if save_trace:

					# set up full path (number of iteration)
					if train_step % 500 == 0:
						num_path = f"{train_step}-{train_step + 499}/"

						if not os.path.exists(output_path + folder_string + num_path):
							os.makedirs(output_path + folder_string + num_path)
					
					if train_step%save_step==0:
						
						# get trace
						layers_as_list = [item.detach().cpu().numpy() for sublist in net.trace for item in sublist]
						X = NN_set_up.array_from_ragged_list(layers_as_list)
						
						# save trace
						np.savetxt(
							f'{output_path}{folder_string}{num_path}'
							f'state_matrix.tstep_{tstep}.S_{S}.iter_{train_step}.csv',
							X, delimiter=',') # fmt='%f'
	
					#print('time set weights total: ', (time()-t3))
					#print('total time rewiring {} s'.format(time()-t2))
	
			else:
	
				#print("\nTraining Time (in minutes) =", (time() - time0) / 60)
				#time1 = time()
	
				'''
				compute training metrics
				'''
				
				# set model to eval mode
				net.eval()
				
				# calculate average training loss / accuracy
				loss = running_loss/n_total_steps
				accuracy = running_correct/running_all
				
				# save loss / accuracy for plotting
				training_loss[e, n_run] = loss
				training_accuracy[e, n_run] = accuracy
				
				# Compute ROC curve and ROC area for each class
				
				# compute useful arrays
				y_true = np.concatenate(y_true[:-1]) # skip last batch bc different size
				# transform to cpu
				y_score = [tensor.cpu() for tensor in y_score]
				y_score = np.concatenate(y_score[:-1])
				
				#y_true = torch.cat(y_true[:-1])
				#y_score = torch.cat(y_score[:-1])
		
				fpr = dict() # false positive rate
				tpr = dict() # true positive rate
				roc_auc = dict()
				
				for i in range(n_classes):
					fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
					roc_auc[i] = auc(fpr[i], tpr[i])
				
				# Compute micro-average ROC curve and ROC area
				fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
				roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
				
				if e == 0: # only on first iteration
					print('-'*39, '{0: ^20}'.format(f'iteration {n_run}'), '-'*39)
					print('{3: ^20}{4: ^20}{0: ^20}{1: ^20}{2: ^20}'.format('training loss', 'training accuracy', 'micro avrg AUC', 'epoch', 'learning rate'))
					print('-'*100)
				print('{3: ^20}{4: ^20.6f}{0: ^20.6f}{1: ^20.6f}{2: ^20.6f}'
					.format(loss, accuracy, roc_auc["micro"], e, lr_scheduler.get_last_lr()[0]))
				
				# save AUROC for plotting
				training_auc_micro[e, n_run] = roc_auc["micro"]
				for i in range(n_classes):
					training_auc_classes[e,i, n_run] = roc_auc[i]
	
				#print("\nCompute training metrics (in minutes) =", (time() - time1) / 60)
				#time2 = time()
				'''
				compute validation metrics
				'''
				
				# validation accuracy 
		
				correct_count, all_count = 0, 0
				#xcorrect_count, xall_count = 0, 0
				
				y_true_validation = []
				y_score_validation = []
				
				for images,labels in valloader:
					
					images = images.view(images.shape[0], -1)
					with torch.no_grad():
						logps = net(images.to(device))
					ps = torch.exp(logps)
					#true_label = labels.numpy()
					true_label = labels.to(device)
					for i, val in enumerate(ps):
						probab = list(val.cpu().numpy())
						pred_label = probab.index(max(probab))
						if true_label[i] == pred_label:
							correct_count += 1
						all_count += 1
					
					y_true_validation.append(label_binarize(labels, classes=[i for i in range(n_classes)]))
					y_score_validation.append(logps)

					# calculate test loss
					test_loss = criterion(logps, labels.to(device))

					test_running_loss += test_loss.item()
				
					#_, predicted = torch.max(logps.data, 1)
					#test_running_correct += (predicted == labels).sum().item()
					#test_running_all += len(labels)
					
				else:

					# calculate average training loss / accuracy
					test_loss_total = test_running_loss/n_total_steps_test
					
					# save loss / accuracy for plotting
					test_loss_save[e, n_run] = test_loss_total

					# calculate average validation accuracy
					validation_accuracy[e, n_run] = correct_count/all_count
					
					# Compute ROC curve and ROC area for each class
					
					# compute useful arrays
					y_true_validation = np.concatenate(y_true_validation[:-1])
					# tranform to cpu
					y_score_validation = [tensor.cpu() for tensor in y_score_validation]
					y_score_validation = np.concatenate(y_score_validation[:-1])
					n_classes = y_true_validation.shape[1]
					
					fpr = dict() # false positive rate
					tpr = dict() # true positive rate
					roc_auc = dict()
		
					for i in range(n_classes):
						fpr[i], tpr[i], _ = roc_curve(y_true_validation[:, i], y_score_validation[:, i])
						roc_auc[i] = auc(fpr[i], tpr[i])
					
					# Compute micro-average ROC curve and ROC area
					fpr["micro"], tpr["micro"], _ = roc_curve(y_true_validation.ravel(), y_score_validation.ravel())
					roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
					
					# save AUROC for plotting
					validation_auc_micro[e, n_run] = roc_auc["micro"]
					for i in range(n_classes):
						validation_auc_classes[e, i, n_run] = roc_auc[i]
	
				# update learning rate according to scheduler
				# lr_scheduler.step()
	
			# save average weights with standard deviation per epoch
			data = torch.nn.utils.parameters_to_vector(net.parameters()).detach()
			average_weights_epoch[e, n_run] = torch.mean(data, dtype=torch.float64)
			stdev_weights_epoch[e, n_run] = torch.std(data, unbiased=False)
			
			# save average activations and standard deviation
			data_act = torch.tensor(NN_rloss.concat_trace(net.trace))
			average_act_epoch[e, n_run] = torch.mean(data_act, dtype=torch.float64)
			stdev_act_epoch[e, n_run] = torch.std(data_act, unbiased=False)
			
			if local_rewiring and save_rewiring and e<stop_rewiring:
				# save rewiring data
				#np.savetxt(f'{output_path}{folder_string}rewiring_activity.csv', np.concatenate(rewiring_activity, axis=1), delimiter=',')
				np.savetxt(f'{output_path}{folder_string}rewiring_oldweights.csv', rewiring_oldweights, delimiter=',')
				np.savetxt(f'{output_path}{folder_string}rewiring_newweights.csv', rewiring_newweights, delimiter=',')
				np.savetxt(f'{output_path}{folder_string}rewiring_preindices.csv', rewiring_preindices, delimiter=',')
				np.savetxt(f'{output_path}{folder_string}rewiring_postindices.csv', rewiring_postindices, delimiter=',')
		
			# save model to file
			net_filename = f'my_mnist_model.iter_{global_run}.epoch_{e+1}.pt' # save as input state for epoch e+1
			torch.save(net, output_path+net_filename)
		
			# plot the activity (stdev)
			fig, ax = plt.subplots()
			for layer, batchneurons in enumerate(net.trace[:-1]):
				ax.scatter([layer]*batchneurons.shape[1], batchneurons.cpu().std(axis=0), alpha=0.01, c='black', s=100)
			ax.set(xlabel='layer', ylabel='activity (stdev)', title='MNIST input')
			fig.savefig(output_path+f'activityMNIST.iter{global_run}.epoch{e+1}.png')
			plt.close()
			
			if local_rewiring:
				# plot the activity (stdev)
				fig, ax = plt.subplots()
				# for layer, batchneurons in enumerate(bornnet.trace[:-1]):
				for layer, batchneurons in enumerate(bornact[:-1]):
					ax.scatter([layer]*batchneurons.shape[1], batchneurons.std(axis=0), alpha=0.01, c='black', s=100)
				ax.set(xlabel='layer', ylabel='activity (stdev)', title='Bornholdt input')
				fig.savefig(output_path+f'activityBORN.iter{global_run}.epoch{e+1}.png')
				plt.close()
		
		 # plot slope of line fit c/LLE plot
		plt.plot(torch.arange(epochs), aline_lle)
		plt.xlabel('epochs')
		plt.ylabel('slope')
		plt.savefig(output_path + f'slope_lle.iter{global_run}.png', bbox_inches='tight')
		plt.close()
		
		# save loss vs weights plot
		# figloss.savefig(output_path + f'loss_vs_weights.iter{global_run}.png', bbox_inches='tight')
		# plt.close()
		
		# save model to list
		net_save[n_run] = net
		
		# save model metrics as pandas dataframe
		df_metrics = pd.DataFrame(data=[
			list(range(epochs)),
			training_loss[:,n_run].flatten().tolist(),
			training_accuracy[:,n_run].flatten().tolist(),
			training_auc_micro[:,n_run].flatten().tolist(),
			validation_accuracy[:,n_run].flatten().tolist(),
			validation_auc_micro[:,n_run].flatten().tolist(),
			test_loss_save[:,n_run].flatten().tolist()
			#[validation_auc_classes[:,i,0] for i,_ in enumerate(validation_auc_classes[0,:,0])]
			]).T
		df_metrics.columns = [
			'epoch',
			'training_loss',
			'training_accuracy',
			'training_auc_micro',
			'validation_accuracy',
			'validation_auc_micro',
			'test_loss'
			]
		df_metrics.to_csv(f'{output_path}model_metrics.iter_{global_run}.csv', sep=',')
		
		tac_flat = np.array([training_auc_classes[i,:,n_run].flatten().tolist() for i in range(epochs)])
		vac_flat = np.array([validation_auc_classes[i,:,n_run].flatten().tolist() for i in range(epochs)])
	
		df_auc_classes = pd.DataFrame(data=np.concatenate([
			np.array(range(epochs))[:,np.newaxis],
			tac_flat,
			vac_flat
			], axis=1))
		df_auc_classes.columns = np.concatenate([
			np.array(['epoch']),
			[f'train auc class {i}' for i in range(n_classes)],
			[f'test auc class {i}' for i in range(n_classes)]
			])
		df_auc_classes.to_csv(f'{output_path}model_auc_classes.iter_{global_run}.csv', sep=',')

		# save weight mean / standard deviation
		df_statistics_weights = pd.DataFrame(data=np.concatenate([
			np.arange(epochs)[:,np.newaxis],
			average_weights_epoch[:,n_run][:,np.newaxis].numpy(),
			stdev_weights_epoch[:,n_run][:,np.newaxis].numpy()
			], axis=1))
		df_statistics_weights.columns = np.concatenate([
			['epoch'],
			['mean'],
			['stdev']
			])
		df_statistics_weights.to_csv(f'{output_path}statistics_weights.iter_{global_run}.csv', sep=',')

		# save activation  mean / standard deviation
		df_statistics_activations = pd.DataFrame(data=np.concatenate([
			np.arange(epochs)[:,np.newaxis],
			average_act_epoch[:,n_run][:,np.newaxis].numpy(),
			stdev_act_epoch[:,n_run][:,np.newaxis].numpy()
			], axis=1))
		df_statistics_activations.columns = np.concatenate([
			['epoch'],
			['mean'],
			['stdev']
			])
		df_statistics_activations.to_csv(f'{output_path}statistics_activations.iter_{global_run}.csv', sep=',')
			
		#with h5py.File(f'{output_path}gaussian_mean.iter_{global_run}.hdf5', 'w') as f:
		#	 dset = f.create_dataset("default", data=gaussian_mean)
		if scale_rconst:
			# save rewiring loss
			np.savetxt(output_path + f'rloss.iter_{global_run}.txt', torch.stack(rloss).tolist())

			# save effective rewiring constant
			np.savetxt(output_path + f'rconst_eff.iter_{global_run}.txt', torch.stack(rconst_eff).tolist())

		# # save lyapunov used for rewiring loss
		# np.savetxt(output_path + f'lyapunov.iter_{global_run}.txt', lyapunov)
		# 
		# # save all lyapunovs
		# with open(output_path + f'all_lyapunovs.iter_{global_run}.pkl', 'wb') as handle:
		# 	pickle.dump(all_lyapunovs, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		# save weight difference SGD and SOC
		np.savetxt(output_path + f'weightdiff_SGD.iter_{global_run}.csv', weightdiff_SGD_save)
		np.savetxt(output_path + f'weightdiff_SOC.iter_{global_run}.csv', weightdiff_SOC_save)
		
		# save the gradients of the loss wrt. weights
		np.savetxt(output_path + f'gradients.iter_{global_run}.csv', gradients_save)
		
		# save the loss of each training iteration
		np.savetxt(output_path + f'loss.iter_{global_run}.csv', loss_save)
		
		# save network output
		#np.savetxt(output_path + f'output.iter_{global_run}.csv', output_save) save each epoch!
		
		print("\nModel Accuracy =", validation_accuracy[-1, n_run].item())
		print()

	print("Number Of Images Tested =", all_count)
	
	shutil.copyfile(sys.argv[0], f'{output_path}{sys.argv[0]}') # save code
	shutil.copyfile('NN_set_up.py', f'{output_path}NN_set_up.py') # save torch network
	shutil.copyfile(f'PARAM_binary_neuron_network.py', f'{output_path}LOG_binary_neuron_network.py') # save logfile


####### END OF NN COMPUTATIONS ########

# in case you just want to visualize metrics, no forward/backward pass through network
# load all previously calculated metrics in corresponding tensors

def load_metrics(
	iterations,
	output_path,
	training_loss,
	training_accuracy,
	training_auc_micro,
	validation_accuracy,
	validation_auc_micro,
	training_auc_classes,
	validation_auc_classes,
	average_weights_epoch,
	stdev_weights_epoch,
	net_save
	):
	
	# load dataframes for each iteration
	for n_run in range(iterations):
		
		# load dataframe and save as arrays
		df_sys_metrics = torch.tensor(pd.read_csv(f'{output_path}model_metrics.iter_{n_run}.csv').iloc[:,2:].values.T)
		df_class_metrics = torch.tensor(pd.read_csv(f'{output_path}model_auc_classes.iter_{n_run}.csv').iloc[:,2:].values)
		df_statistics_weights = torch.tensor(pd.read_csv(f'{output_path}statistics_weights.iter_{n_run}.csv').iloc[:,2:].values)
		
		# load network
		net_save[n_run] = torch.load(f'{output_path}my_mnist_model.iter_{n_run}.pt', map_location=torch.device('cpu'))
		
		# fill arrays accordingly
		
		(
		training_loss[:,n_run],
		training_accuracy[:,n_run],
		training_auc_micro[:,n_run],
		validation_accuracy[:,n_run],
		validation_auc_micro[:,n_run]
		) = df_sys_metrics
		
		nr_classes = int(df_class_metrics.shape[1]/2)
		training_auc_classes[:,:,n_run] = df_class_metrics[:,:nr_classes]
		validation_auc_classes[:,:,n_run] = df_class_metrics[:,nr_classes:]
		
		average_weights_epoch[:,n_run] = df_statistics_weights[:,0]
		stdev_weights_epoch[:,n_run] = df_statistics_weights[:,1]
		
	return training_loss, training_accuracy, training_auc_micro, validation_accuracy, validation_auc_micro, training_auc_classes, validation_auc_classes, average_weights_epoch, stdev_weights_epoch, net_save

if pars_arg.no_comp:
	print('No computations mode. Load precalculated metrics.')
	
	# in no comp mode, save to place where data was fetched
	output_path = parent_dir + output_path
	
	# load metrics in preallocated tensors
	(
	training_loss,
	training_accuracy,
	training_auc_micro,
	validation_accuracy,
	validation_auc_micro,
	training_auc_classes,
	validation_auc_classes,
	average_weights_epoch,
	stdev_weights_epoch,
	net_save
	) = load_metrics(
			iterations,
			output_path,
			training_loss,
			training_accuracy,
			training_auc_micro,
			validation_accuracy,
			validation_auc_micro,
			training_auc_classes,
			validation_auc_classes,
			average_weights_epoch,
			stdev_weights_epoch,
			net_save
			)

def view_training(training_loss, training_accuracy, training_auc_micro, training_auc_classes):
	'''
	function for viewing training metrics.
	'''
	
	fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(ncols=4, nrows=3, figsize=(16,12)) 
	
	# plot loss, accuracy, and (micro) AUROC
	
	# loss
	ax1.plot(training_loss, label=[f'run {i}' for i in range(iterations)])
	ax1.set_title('Training loss')
	ax1.set_ylabel('loss')
	ax1.set_aspect(1./ax1.get_data_ratio(), adjustable='box')
	
	# accuracy
	ax2.plot(training_accuracy)
	ax2.set_title('Training accuracy')
	ax2.set_ylabel('accuracy')
	ax2.set_aspect(1./ax2.get_data_ratio(), adjustable='box')

	# micro auc
	ax3.plot(training_auc_micro)
	ax3.set_title('training micro-avrg AUROC score')
	ax3.set_ylabel('AUROC')
	ax3.set_aspect(1./ax3.get_data_ratio(), adjustable='box')

	# class auc
	nr_classes = training_auc_classes.shape[1]
	ax4.plot(training_auc_classes[:,:,0], label=[f'class {i}' for i in range(nr_classes)])
	ax4.set_title('training class AUROC score first iteration')
	ax4.set_ylabel('AUROC')
	ax4.set_aspect(1./ax4.get_data_ratio(), adjustable='box')
	ax4.legend()
	
	# calculate mean and standard deviation
	
	# mean over all iterations/classes 
	mean_loss = torch.mean(training_loss, axis=1)
	mean_acc = torch.mean(training_accuracy, axis=1)
	mean_auc_micro = torch.mean(training_auc_micro, axis=1)
	mean_auc_classes = torch.mean(training_auc_classes[:,:,0], axis=1)
	
	# standard deviations over all iterations/classes
	std_loss = torch.std(training_loss, axis=1) 
	std_acc = torch.std(training_accuracy, axis=1)
	std_auc_micro = torch.std(training_auc_micro, axis=1)
	std_auc_classes = torch.std(training_auc_classes[:,:,0], axis=1)
	
	# plot mean and standard deviation
	
	ax5. plot(mean_loss)
	ax5.set_ylabel('mean loss')
	ax5.set_aspect(1./ax5.get_data_ratio(), adjustable='box')

	ax6.plot(mean_acc)
	ax6.set_ylabel('mean accuracy')
	ax6.set_aspect(1./ax6.get_data_ratio(), adjustable='box')
	
	ax7.plot(mean_auc_micro)
	ax7.set_ylabel('mean AUROC')
	ax7.set_aspect(1./ax7.get_data_ratio(), adjustable='box')
	
	ax8.plot(mean_auc_classes)
	ax8.set_ylabel('mean AUROC')
	ax8.set_aspect(1./ax8.get_data_ratio(), adjustable='box')
	
	ax9.plot(std_loss)
	ax9.set_xlabel('epoch')
	ax9.set_ylabel('standard deviation')
	ax9.set_aspect(1./ax9.get_data_ratio(), adjustable='box')

	ax10.plot(std_acc)
	ax10.set_xlabel('epoch')
	ax10.set_ylabel('standard deviation')
	ax10.set_aspect(1./ax10.get_data_ratio(), adjustable='box')
	
	ax11.plot(std_auc_micro)
	ax11.set_xlabel('epoch')
	ax11.set_ylabel('standard deviation')
	ax11.set_aspect(1./ax11.get_data_ratio(), adjustable='box')
	
	ax12.plot(std_auc_classes)
	ax12.set_xlabel('epoch')
	ax12.set_ylabel('standard deviation')
	ax12.set_aspect(1./ax12.get_data_ratio(), adjustable='box')
	
	#plt.legend()
	fig.tight_layout() # set the right distances btw subplots
	
	fig.savefig(f"{output_path}metrics_training_mnist_digit_recognition.png")
	#np.savetxt(f"{output_path}metrics_training_mnist_digit_recognition.csv", delimiter=',')

view_training(training_loss, training_accuracy, training_auc_micro, training_auc_classes)

def view_validation(validation_accuracy, validation_auc_micro, validation_auc_classes):
	'''
	function for viewing test metrics.
	'''
	
	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(figsize=(12,12), ncols=3, nrows=3)
	
	# plot accuracies and AUROCs
	
	# accuracy
	ax1.plot(validation_accuracy)
	ax1.set_title('validation accuracy')
	ax1.set_ylabel('accuracy')
	ax1.set_aspect(1./ax1.get_data_ratio(), adjustable='box')

	# micro-AUROC
	ax2.plot(validation_auc_micro)
	ax2.set_title('validation micro-avrg AUROC score')
	ax2.set_ylabel('AUROC')
	ax2.set_aspect(1./ax2.get_data_ratio(), adjustable='box')

	# class AUROC
	nr_classes = validation_auc_classes.shape[1]
	ax3.plot(validation_auc_classes[:,:,0], label=[f'class {i}' for i in range(nr_classes)])
	ax3.set_title('validation class AUROC score first iteration')
	ax3.set_ylabel('AUROC')
	ax3.set_aspect(1./ax3.get_data_ratio(), adjustable='box')
	ax3.legend()

	# calculate averages and standard deviations

	# calculate averages over all iterations/classes of test metrics
	mean_acc = torch.mean(validation_accuracy, axis=1)
	mean_auc_micro = torch.mean(validation_auc_micro, axis=1)
	mean_auc_classes = torch.mean(validation_auc_classes, axis=1)
	
	# calculate standard deviations between iterations/classes
	std_acc = torch.std(validation_accuracy, axis=1)
	std_auc_micro = torch.std(validation_auc_micro, axis=1)
	std_auc_classes = torch.std(validation_auc_classes[:,:,0], axis=1)
	
	# plot mean and standard deviation of all metrics

	ax4.plot(mean_acc)
	ax4.set_ylabel('mean accuracy')
	ax4.set_aspect(1./ax4.get_data_ratio(), adjustable='box')

	ax5.plot(mean_auc_micro)
	ax5.set_ylabel('mean AUROC')
	ax5.set_aspect(1./ax5.get_data_ratio(), adjustable='box')

	ax6.plot(mean_auc_classes)
	ax6.set_ylabel('mean AUROC')
	ax6.set_aspect(1./ax6.get_data_ratio(), adjustable='box')

	ax7.plot(std_acc)
	ax7.set_xlabel('epoch')
	ax7.set_ylabel('standard deviation')
	ax7.set_aspect(1./ax7.get_data_ratio(), adjustable='box')

	ax8.plot(std_auc_micro)
	ax8.set_xlabel('epoch')
	ax8.set_ylabel('standard deviation')
	ax8.set_aspect(1./ax8.get_data_ratio(), adjustable='box')

	ax9.plot(std_auc_classes)
	ax9.set_xlabel('epoch')
	ax9.set_ylabel('standard deviation')
	ax9.set_aspect(1./ax9.get_data_ratio(), adjustable='box')

	#plt.legend()
	fig.tight_layout()
 
	fig.savefig(f"{output_path}metrics_validation_mnist_digit_recognition.png")
	#np.savetxt(f"{output_path}metrics_validation_mnist_digit_recognition.csv", delimiter=',')

view_validation(validation_accuracy, validation_auc_micro, validation_auc_classes)

# testing & evaluation

# plot average weights
def view_weights_epochs(weights, std, global_run):
	'''
	This function plots the average weights with standard deviation for each epoch and iteration.
	The tensors weights and std contains the weights and standard deviation for corresponding epochs (dim0)
	and iterations (dim1).
	'''

	weights = weights.cpu().data.numpy()
	std = std.cpu().data.numpy()

	# plot average and standard deviation

	fig, (ax1, ax2) = plt.subplots(2)

	ax1.plot(weights)
	#ax1.x_label('epoch')
	ax1.set_ylabel(r'average weights $\bar{w}$')

	ax2.plot(std)
	ax2.set_xlabel('epoch')
	ax2.set_ylabel(r'standard deviation $\sigma$')

	fig.savefig(f'{output_path}average_weights_per_epoch.iter_{global_run}.png')
	

view_weights_epochs(average_weights_epoch, stdev_weights_epoch, global_run)

# plot average activations
def view_activations_epochs(activations, std, global_run):
	'''
	This function plots the average activation with standard deviation for each epoch and iteration.
	The tensors weights and std contains the weights and standard deviation for corresponding epochs (dim0)
	and iterations (dim1).
	'''

	activations = activations.cpu().data.numpy()
	std = std.cpu().data.numpy()

	# plot average and standard deviation

	fig, (ax1, ax2) = plt.subplots(2)

	ax1.plot(activations)
	#ax1.x_label('epoch')
	ax1.set_ylabel(r'average activation $\bar{a}$')

	ax2.plot(std)
	ax2.set_xlabel('epoch')
	ax2.set_ylabel(r'standard deviation $\sigma$')

	fig.savefig(f'{output_path}average_activation_per_epoch.iter_{global_run}.png')
	

view_activations_epochs(average_act_epoch, stdev_act_epoch, global_run)

# plot weight histograms
def view_weights_histogram(net_save, output_path, iterations):
	
	# plot histograms
	iterations = len(net_save)
	fig, ax_iter = plt.subplots(ncols=iterations, figsize=(iterations*4,4))
	
	for i, ax in enumerate(ax_iter):
		# get weights from network instance
		weights = torch.nn.utils.parameters_to_vector(net_save[i].parameters()).detach().numpy()
		
		# draw histogram
		sb.histplot(data=weights, ax=ax, stat='probability', color='black')
		ax.set_xlabel(r'coupling strength $w_{ij}$')
		ax.set_title(f'iteration {i}')
		ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')

	fig.tight_layout()
	fig.savefig(f'{output_path}weight_histogram.png')

# only calculate if iterations are not clustered
if not pars_arg.cluster_iterations:
	view_weights_histogram(net_save, output_path, iterations)

# show examplary result
def view_classify(img1, img2, img3, ps1, ps2, ps3):
	'''
	Function for viewing an image and it's predicted classes.
	'''
	ps1 = ps1.cpu().data.numpy().squeeze()
	ps2 = ps2.cpu().data.numpy().squeeze()
	ps3 = ps3.cpu().data.numpy().squeeze()
	
	fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(figsize=(6,9), ncols=2, nrows=3)
	
	ax1.imshow(img1.resize_(1, 28, 28).numpy().squeeze(), cmap='Greys')
	ax1.axis('off')
	ax1.set_title('Image')
	ax2.barh(np.arange(10), ps1)
	ax2.set_aspect(0.1)
	ax2.set_yticks(np.arange(10))
	ax2.set_yticklabels(np.arange(10))
	ax2.set_title('Class Probability')
	ax2.set_xlim(0, 1.1)
	
	ax3.imshow(img2.resize_(1, 28, 28).numpy().squeeze(), cmap='Greys')
	ax3.axis('off')
	ax4.barh(np.arange(10), ps2)
	ax4.set_aspect(0.1)
	ax4.set_yticks(np.arange(10))
	ax4.set_yticklabels(np.arange(10))
	ax4.set_xlim(0, 1.1)
	
	ax5.imshow(img3.resize_(1, 28, 28).numpy().squeeze(), cmap='Greys')
	ax5.axis('off')
	ax6.barh(np.arange(10), ps3)
	ax6.set_aspect(0.1)
	ax6.set_yticks(np.arange(10))
	ax6.set_yticklabels(np.arange(10))
	ax6.set_xlim(0, 1.1)
	plt.tight_layout()
	
	fig.savefig(f"{output_path}mnist_predict.png")
		
#images, labels = next(iter(valloader))
#
#img1 = images[0].view(1, 784)
#with torch.no_grad():
#	 logps = net(img1.to(device))
#
#ps1 = torch.exp(logps)
#probab1 = list(ps1.cpu().numpy()[0])
##print("Predicted Digit =", probab1.index(max(probab1)))
#
#img2 = images[1].view(1, 784)
#with torch.no_grad():
#	 logps = net(img2.to(device))
#
#ps2 = torch.exp(logps)
#probab2 = list(ps2.cpu().numpy()[0])
##print("Predicted Digit =", probab2.index(max(probab2)))
#
#img3 = images[2].view(1, 784)
#with torch.no_grad():
#	 logps = net(img3.to(device))
#
#ps3 = torch.exp(logps)
#probab3 = list(ps3.cpu().numpy()[0])
##print("Predicted Digit =", probab3.index(max(probab3)))
#
#view_classify(img1.view(1, 28, 28), img2.view(1, 28, 28), img3.view(1, 28, 28), ps1, ps2, ps3)

# validation accuracy 

#correct_count, all_count = 0, 0
#
#y_true_validation = []
#y_score_validation = []
#validation_auc_micro = np.zeros(epochs)
#validation_auc_classes = np.zeros((epochs, n_classes))
#
#for images,labels in valloader:
#	 for i in range(len(labels)):
#		 img = images[i].view(1, 784)
#		 # Turn off gradients to speed up this part
#		 with torch.no_grad():
#			 logps = net(img.to(device))
#		 
#		 # Output of the network are log-probabilities, need to take exponential for probabilities
#		 ps = torch.exp(logps)
#		 probab = list(ps.cpu().numpy()[0])
#		 pred_label = probab.index(max(probab))
#		 true_label = labels.numpy()[i]
#		 if(true_label == pred_label):
#		   correct_count += 1
#		 all_count += 1
#
#		 y_true_validation.append(label_binarize(labels, classes=[i for i in range(n_classes)]))
#		 y_score_validation.append(logps)
#	 
#else:
#	 # Compute ROC curve and ROC area for each class
#	 
#	 # compute useful arrays
#	 y_true_validation = np.concatenate(y_true_validation[:-1])
#	 y_score_validation = np.concatenate(y_score_validation[:-1])
#	 n_classes = y_true_validation.shape[1]
#	 
#	 fpr = dict() # false positive rate
#	 tpr = dict() # true positive rate
#	 roc_auc = dict()
#	 
#	 for i in range(n_classes):
#		 fpr[i], tpr[i], _ = roc_curve(y_true_validation[:, i], y_score_validation[:, i])
#		 roc_auc[i] = auc(fpr[i], tpr[i])
#	 
#	 # Compute micro-average ROC curve and ROC area
#	 fpr["micro"], tpr["micro"], _ = roc_curve(y_true_validation.ravel(), y_score_validation.ravel())
#	 roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#	 
#	 # save AUROC for plotting
#	 validation_auc_micro[e] = roc_auc["micro"]
#	 for i in range(n_classes):
#		 validation_auc_classes[e,i] = roc_auc[i]

print('output path:', output_path)

end_time = time()
print('Program ended at {} after {} min.'.format(ctime(), round((end_time-start_time)/60, 3)))
