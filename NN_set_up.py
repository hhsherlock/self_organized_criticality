import torch
import numpy as np
import copy
import math
from numba import jit, njit, prange
from numba.typed import List
from numba.types import uint
from time import time

#from PARAM_binary_neuron_network import batchsize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # check available device

def activation_meisel20(x, p_ne=1, device=device):
	'''
	Activation function used in Meisel20.
	Reference: www.pnas.org/cgi/doi/10.1073/pnas.1911461117
	'''
	
	prob = torch.clamp(p_ne*x, min=0, max=p_ne)#.to(device) # probability to fire
	x = (torch.rand(prob.size()).to(device) < prob).float() # .to(device)
	# return x
	return prob

def activation_tanh(x, a=1.7159, b=0.6666): #  a=1.7159, b=0.6666):
	'''
	Activation function used in Cireşan12.
	DOI: 10.1007/978-3-642-35289-8_31
	'''
	
	x = a*torch.tanh(b*x)
	#x = (torch.rand(prob.size()).to(device) < prob).to(torch.float64).to(device)
	return x

class ScaledTanh(torch.nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh(0.6666 * x)

def activation_lintanh(x, a=1.7159, b=1.4):
	'''
	Partially linear function similar to the hyperbolic
	tangent activation function used in Cireşan12.
	'''
	
	y = torch.clamp(b*x, min=-a, max=a)
	return y

def calc_jacobian_tanh(weights, x, a=1.7159, b=0.6666):
	"""Calculates the jacobian matrix. Double transpose so each row 
	is multiplied with the same element of the states vector
	"""

	jacobi = (a*b*weights.T * (torch.cosh(b*torch.matmul(weights,x)))**(-2)).T

	return jacobi

def calc_jacobian_meisel20(weights, x, p=1.0):
	"""Calculates the jacobian matrix of e network with chrstians
	transfer function. Additional dimension in last term so all rows
	of the weight matrix get scaled with the same term
	"""

	neuron_input = torch.matmul(weights, x)
	jacobi = weights * ((0<neuron_input)*(neuron_input<1))[:,None]*p

	return jacobi

class InputTransform(torch.nn.Module):
	def __init__(self, input_features, learnable=True):
		super().__init__()
		
		if learnable:
			# register as learnable parameter
			self.scale = torch.nn.Parameter(torch.ones(input_features))
			self.shift = torch.nn.Parameter(torch.zeros(input_features))
		
		else:
			# register as non-learnable tensors
			self.register_buffer('scale', torch.ones(input_features))
			self.register_buffer('shift', torch.zeros(input_features))
	
	def forward(self, x):
		
		# return (x-x.mean(axis=0))/x.std(axis=0) * self.scale + self.shift
		return x * self.scale + self.shift

class net_init(torch.nn.Module):
	def __init__(self, n_l, model, activity_threshold, bstep, batch_norm, bn_fixed, gamma, beta, input_transform, it_fixed, scale, shift):
		super().__init__()

		# layers
		self.linears = torch.nn.ModuleList(
			[torch.nn.Linear(n_l[i],n_l[i+1], bias=False) for i in range(len(n_l)-1)])
		
		# batch norm as described in https://doi.org/10.48550/arXiv.1502.03167
		self.batchnorm = batch_norm
		self.bn = torch.nn.ModuleList(
			[torch.nn.BatchNorm1d(n_l[i+1]) for i in range(len(n_l)-1)])
		
		# neuron input transform (scale and shift)
		self.inputtransform = input_transform
		self.it = torch.nn.ModuleList(
			[InputTransform(n_l[i+1], learnable=not it_fixed) for i in range(len(n_l)-1)])
		
		# iterate batchnorm layers
		for bn_layer in self.bn:
			
			# set intial values for gamma and beta
			bn_layer.weight.data.fill_(gamma)
			bn_layer.bias.data.fill_(beta)
			
			# trainable or not
			bn_layer.weight.requires_grad = not bn_fixed 
			bn_layer.bias.requires_grad = not bn_fixed
		
		# iterate neuron input scaling layers
		for it_layer in self.it:
			
			# set initial values of scaling and shift
			it_layer.scale.data.fill_(scale)
			it_layer.shift.data.fill_(shift)
		
		# activation last layer
		self.logsoftmax = torch.nn.LogSoftmax(dim=1) # multi-class classification
		self.sigmoid = torch.nn.Sigmoid() # binary classification

		# criticality measure rewiring loss
		self.thresh = torch.tensor(activity_threshold)
		self.bstep = bstep
		self.layer_maxsize = max(n_l) # max nr of nodes in layer
		self.nlayers = len(self.linears)+1 # number of layers
		self.nl = n_l # network structure

		# which activation function?
		if model=='meisel20':
			self.activation = activation_meisel20
		elif model=='relu':
			self.activation = torch.nn.ReLU()
		# elif model=='tanh':
		# 	# self.activation = activation_tanh
		# 	self.activation = ScaledTanh()
		elif model=='lintanh':
			self.activation = activation_lintanh
		else: 
			self.activation = ScaledTanh()

		# which activation in last layer?
		if self.linears[-1].out_features == 1:
			# activate output layer with sigmoid
			self.lastlayer = torch.nn.Sigmoid()
		else:
			# activate output layers with softmax
			self.lastlayer = torch.nn.LogSoftmax(dim=1)

		# which jacobian function (lle calc)
		if model=='meisel20':
			self.jacobian = calc_jacobian_meisel20
		if model=='tanh':
			self.jacobian = calc_jacobian_tanh

	def forward(self, x, temptrace=False):#, p_ne, device):
		'''
		The forward pass receives a Tensor containing the input and returns
		a Tensor containing the output.
		'''
		
		# empty all traces before forward
		self.pretrace = []
		savetrace = []
		
		# choose trace to save forward
		# self.trace = trace2 if temptrace==True else trace1
		
		self.pretrace.append(x.detach())
		savetrace.append(x.detach()) # include input layer configuration
		
		self.batchsize = x.shape[0]
		
		for l, layer in enumerate(self.linears[:-1]):
			
			# make sure input tensor is flattened
			x = x.view(x.shape[0], -1)
			
			# linear mapping to next layer
			x = layer(x)
			
			# apply batch norm
			if self.batchnorm: x = self.bn[l](x)
			
			# apply input transform
			if self.inputtransform: x = self.it[l](x)
			
			# save neuron inputs
			self.pretrace.append(x.detach())
			
			# apply transfer function
			x = self.activation(x)

			# layers to trace
			savetrace.append(x.detach())

		# send to last layer
		x = self.linears[-1](x)
		
		# apply batch norm
		if self.batchnorm: x = self.bn[-1](x)
		
		# apply neuron input transform
		if self.inputtransform: x = self.it[-1](x)
		
		# save neuron input last layer
		self.pretrace.append(x.detach())
		
		# activate last layer
		x = self.lastlayer(x)

		# output layer to trace
		savetrace.append(x.detach())
		
		# save to corresponding trace variable
		if temptrace == True:
			self.temptrace = savetrace
		else:
			self.trace = savetrace

		return x
	
	def born_wup_LOOP(self, l, layer, x):
		
		# non-vectorized version 
		
		# weight adjustment based on activity
		with torch.no_grad():
			
			# how many neurons are rewired?
			nneurons = int(layer.weight.shape[1] * 0.1)
			
			# how many links per neuron?
			nlinks = int(self.linears[l-1].weight.shape[1] * 0.1)
			
			# draw subset of neurons that are rewired
			rand_neurons = torch.randperm(x.shape[1], device=x.device)[:nneurons]
			
			# are these neurons active or inactive?
			active = x[:,rand_neurons].std(axis=0) > self.thresh
				
			# weights that are changed
			wvals = lambda nrn, lnks: self.linears[l-1].weight[nrn, lnks]
			
			# iterate subset of neurons 
			for randn, act in zip(rand_neurons, active):
				
				# is the neurons active or not?
				if act:
					
					# only choose non-zero links
					nonzero_links = torch.where(self.linears[l-1].weight[randn,:] != 0)[0]
					permute = nonzero_links[torch.randperm(len(nonzero_links))[:nlinks]]
					
					# change weights active neuron
					self.linears[l-1].weight[randn, permute] = (wvals(randn, permute).abs() - self.bstep).clamp(0,None) * wvals(randn, permute).sign()
					
				else: # inactive
					
					# choose links
					permute = torch.randperm(self.linears[l-1].weight.shape[1])[:nlinks]
					
					# are weights zero?
					zeroweights = wvals(randn, permute) == 0
					
					# weight update zero links
					wup_zero = zeroweights * self.bstep * torch.randn(len(zeroweights), device=x.device).sign()
					
					# weight update non-zero links
					wup_nonzero = (~zeroweights) * (wvals(randn, permute).abs() + self.bstep) * wvals(randn, permute).sign()
					
					# change weights inactive neuron
					self.linears[l-1].weight[randn, permute] = wup_zero + wup_nonzero
	
	def born_wup(self, l, x, fraction_neurons, fraction_links, step_scaling=1):
		"""Vectorized implementation of Bornoldt, Rohlf 2000 weight update rule
		
		ARG
		- l ... layer the neurons live in 
		- x ... neuron states in layer l
		
		RETURN
			updated weights in network instance
		"""
		
		# step_scaling = 1 # 2**(l-1)
		# print('layer =', l)
		
		with torch.no_grad():
			
			# parameters
			nneurons = int(math.ceil(x.shape[1] * fraction_neurons))
			nlinks = int(math.ceil(self.linears[l-1].weight.shape[1] * fraction_links))
			
			# select subset of neurons for rewiring
			rand_neurons = torch.randperm(x.shape[1], device=x.device)[:nneurons]
			# print('rand_neurons =', rand_neurons)
			
			# mask active/inactive neurons
			active_neurons = x[:, rand_neurons].std(axis=0) > self.thresh
			
			# separate active and inactive neurons
			active_indices = rand_neurons[active_neurons]
			inactive_indices = rand_neurons[~active_neurons]
			
			# process active neurons
			if len(active_indices) > 0:
				
				# mask nonzero weights
				nonzero_links = (self.linears[l-1].weight[active_indices] != 0).float()
				
				# choose nonzero
				selected_links = torch.multinomial(nonzero_links, nlinks, replacement=False)
				# print('selected_links =', selected_links)
				
				# process weights
				selected_weights = self.linears[l-1].weight[active_indices[:, None], selected_links]
				adjusted_weights = (selected_weights.abs() - self.bstep*step_scaling).clamp(0, None) * selected_weights.sign()
				
				# save in instance
				self.linears[l-1].weight[active_indices[:, None], selected_links] = adjusted_weights
				
				# # change weight in-place
				# weight_signs = self.linears[l-1].weight[active_indices[:, None], selected_links].sign()
				# self.linears[l-1].weight[active_indices[:, None], selected_links].abs_().sub_(self.bstep*step_scaling).clamp_(min=0, max=None).mul_(weight_signs)
			
			# process inactive neurons
			if len(inactive_indices) > 0:
				
				# choose from all weights
				all_links = torch.full_like(self.linears[l-1].weight[inactive_indices], 1.)
				random_links = torch.multinomial(all_links, nlinks, replacement=False)
				# print('random_links =', random_links)
				
				# process weights
				selected_weights = self.linears[l-1].weight[inactive_indices[:, None], random_links]
				zero_weights = selected_weights == 0
				wup_zero = zero_weights * self.bstep * step_scaling * torch.randn_like(selected_weights).sign()
				wup_nonzero = (~zero_weights) * (selected_weights.abs() + self.bstep*step_scaling) * selected_weights.sign()
				
				# save in instance
				self.linears[l-1].weight[inactive_indices[:, None], random_links] = wup_zero + wup_nonzero
				
				# # change weights in-place
				# zero_weights = self.linears[l-1].weight[inactive_indices[:, None], random_links] == 0
				# self.linears[l-1].weight[inactive_indices[:, None], random_links][zero_weights].uniform_(-1,1).sign_().mul_(self.bstep * step_scaling)
				# weight_sign = self.linears[l-1].weight[inactive_indices[:, None], random_links][~zero_weights].sign()
				# self.linears[l-1].weight[inactive_indices[:, None], random_links][~zero_weights].abs_().add_(self.bstep*step_scaling).mul_(weight_sign)
				
	def bornholdt(self, x, frac_neurons, frac_links):
		"""This function updates the weights according to the rules outlined
		in Bornholdt, Rohlf 2000.
		"""
		
		for l, _ in enumerate(self.linears): # dont skip output layer
			
			# make shure inputs are flat
			x = x.view(x.shape[0], -1)
			
			# save layer l 
			# x_current = x
			
			# forward pass
			x = self.linears[l](x)
			
			# save layer l+1
			# x_next = x
			
			# calc sigma, average over neurons and batch
			# sigma = x_next.abs().mean() / x_current.abs().mean()
			
			# updates incoming weights
			self.born_wup(l+1, x, frac_neurons, frac_links)
			x = self.activation(x)

			# print('rewired layer', l, '\t activity =', mean_layer_activity.item())
				
			# # only rewire until critical threshold not reached in layer
			# mean_layer_activity = x.std(axis=0).mean()
			# if not (mean_layer_activity.isclose(self.thresh, rtol=1e-1, atol=0)).all():
			# 	break
			
			# # dont rewire if layer not critical
			# if l>0 and not sigma.round(decimals=1)==1: 
			# 	print('layer', l, 'stop, \t sigma =', sigma.item())
			# 	break
	
	def adjust_signs(self, Q, R):
		"""
		Adjust the signs of the columns of Q and the corresponding rows of R
		such that all diagonal elements of R are positive.
		"""
		# Diagonal elements of R
		diag = torch.diag(R)
		
		# Determine the signs for each diagonal entry
		signs = torch.sign(diag)
		signs[signs == 0] = 1  # Replace zeros with ones to avoid multiplying by zero
		
		# Adjust columns of Q
		Q *= signs.unsqueeze(0).expand_as(Q)
		
		# Adjust rows of R
		R *= signs.unsqueeze(1).expand_as(R)
		
		return Q, R


	def lle(self):
		"""This function calculates the largest lyapunov exponents using
		the approach in Kondo 2017.
		"""

		with torch.no_grad():

			# calc states for single neuron input
			# rand_node = torch.randperm(self.nl[0])[0]
			# input_vec = torch.zeros(self.nl[0])
			# input_vec[rand_node] = torch.rand(1)*2-1 # random number in (-1,1]
			# self.forward(input_vec[None,:]) # saves trace
			# self.forward(x, temptrace=True)
			
			# input_nodes = self.linears[0].weight.shape[1] # number of input nodes
			# input_vec = torch.rand(input_nodes)*2-1 * 1e-9
			# self.forward(input_vec[None,:].to(device), temptrace=True)
			
			# get activations from trace
			# activations = self.temptrace
			activations = self.trace
			# torch.save(activations, output_path + 'llecalc.activations_epoch{e}.pt')

			nlayers = len(activations) # number of layers

			batchsize = activations[0].shape[0] # batchsize

			nneurons = activations[0].shape[1] # number of neurons in input layer 

			# tensor to hold unitspheres 
			# usphere_init = torch.eye(nneurons)[None,...].repeat_interleave(batchsize, axis=0) # provide init for each sample
			usphere_init = torch.eye(nneurons)[None,...].repeat_interleave(batchsize, axis=0) # changed
			usphere = torch.full((batchsize, self.nl.max(), self.nl.max()), torch.nan).to(device) # filled with nans
			usphere[:, :usphere_init.shape[1], :usphere_init.shape[2]] = usphere_init # set for first samples

			# init vector holding data later
			growthrate = torch.full((nlayers-1, batchsize, self.nl[:-1].max()), torch.nan) # skip output layer
			jacobi_sample0_neuron0 = torch.full((nlayers-1, self.nl.max(), self.nl.max()), torch.nan)
			# lyap_layer = torch.full((nlayers-2, batchsize), torch.nan)
			lyap_layer = torch.full((nlayers-1, batchsize, self.nl.max()), torch.nan)
			
			Rdiag = torch.full((nlayers-1, batchsize, self.nl[:-1].max()), torch.nan)

			# get weights in all layers
			weights = list(self.parameters())
			
			# allocate variable that saves the first layer to saturate
			layer_saturate = -1

			# iterate all layers
			for layer, activations_layer in enumerate(activations[:-1]): # skip output layer
				
				# if layer > 1: break # we only use first layer anyway
				# if layer > 0: break # we only use first layer anyway
				
				# check if any neuron in a layer / batch starts saturating
				if layer_saturate==-1 and layer>0 and (activations_layer>1e-3).any(): layer_saturate = layer-1
				
				# get next layer activations
				activations_nextlayer = activations[layer+1]
				
				# iterate all samples
				for sample, activations_sample in enumerate(activations_layer):

					# get jacobian matrix
					jacobi = self.jacobian(weights[layer], activations_sample).to(device)
					if sample==0:
						jacobi_sample0_neuron0[layer, :jacobi.shape[0], :jacobi.shape[1]] = jacobi

					# get unitsphere this sample
					usphere_sample = usphere[sample, :activations_sample.shape[0], :activations_sample.shape[0]]
					# map unit sphere to next timestep with jacobian
					usphere_map = jacobi.matmul(usphere_sample)

					# get growthrate from new sphere
					gr = torch.linalg.vector_norm(usphere_map, axis=0)
					growthrate[layer, sample, :gr.shape[0]] = gr

					# orthogonalize unitsphere
					Q,R = torch.linalg.qr(usphere_map, mode='complete')
					# Q,R = torch.linalg.qr(usphere_map, mode='reduced') # changed
					# Q,R = self.adjust_signs(Q,R) # changed
					
					# save diagonal elements of R
					# Rdiag[layer, sample, :len(torch.diag(R))] = torch.diag(R)

					# save orthonormalized unit sphere for this sample
					usphere[sample, :Q.shape[0], :Q.shape[1]] = Q

					# calc lyapunov for each layer
					# lyap_layer[layer, sample] = torch.log(gr[0]) # apply scaling here
					# lyap_layer[layer, sample] = torch.log(gr.max()) # apply scaling here, take maximum growthrate
					lyap_layer[layer, sample, :gr.shape[0]] = torch.log(gr) # apply scaling here, take all growthrates

			# calculate lyapunovs, max over layer and dimensions, mean over samples
			# lyap = torch.log(growthrate.nan_to_num()).max(axis=0).values.max(axis=-1).values.mean() # maybe use indices 0 and not max?
			# lyap1 = torch.log(growthrate[1:,:,0]).max(axis=0).values.mean() # use first index, maximum over layers and mean over samples
			# lyap = lyap_layer[1:,:].max(axis=0).values.mean() # skip input layer
			# lyap = lyap_layer[1:,:].mean(axis=0).mean() # skip input layer, mean over layer and samples
			# lyap = lyap_layer[1:,:,:].mean(axis=1).flatten() # mean over all samples
			# lyap = torch.log(growthrate[1:-1,:,:].nanmean(axis=1)).flatten() # /len(self.nl[1:-1])
			lyap = torch.log(growthrate[1:layer_saturate,:,:].nanmean(axis=1)).flatten() # only take LEs from first layer to last layer not saturating
			lyap = lyap[~torch.isnan(lyap)] # get rid of nan values
			# lyap = torch.log(growthrate[0,:,:].nanmean(axis=0)).flatten() # only take LEs from first layer
			# lyap = torch.log(growthrate[2:-1,:,:].nanmean(axis=1)).flatten() # only take LEs from first layer
			# print("growthrate:", growthrate.nanmean(axis=1))
			# print("lyap:", lyap)
			
			# calculate lyapunov exponents from diagonal elements of R
			lyap_new = Rdiag.log().nanmean(axis=1).flatten()
			lyap_new = lyap_new[~torch.isnan(lyap_new)]
			
			return growthrate, lyap, activations, jacobi_sample0_neuron0, layer_saturate, Rdiag

def floor_to_decimal(number, decimal_place):
	"""This function floors the number to the given
	decimal place. Used for computing the correct number path 
	in net simulations.
	
	ARGS:
	number -- (float) The number to be floored
	decimal_place -- (float) The decimal place to which number is floored
	
	RETURN:
	-- (float) floored number
	"""

	return int(math.floor(number / decimal_place)) * decimal_place

def custom_weights_to_tensor(W, net):
	'''Inputs a sparse coupling matrix and an instantiated network, 
	returns net with customized weights'''
	
	# construct array representing network structure (n_l) and node indices
	#net_struc = np.zeros(len(net.linears)+2, int) # include input layer and concatenated zero
	#for i, module in enumerate(net.linears):
	#	 net_struc[i+1] = module.in_features
	#net_struc[-1] = net.linears[-1].out_features
	#layer_indices = np.cumsum(net_struc)
	
	# tranform vectors to parameters, convertto csr matrix to get ordering right
	torch.nn.utils.vector_to_parameters(torch.tensor(W.tocsr().data), net.parameters())
	
	#for layer in range(len(net.linears)):
	#	 #outgoing = range(layer_indices[layer], layer_indices[layer+1]) # presynaptic
	#	 #incoming = range(layer_indices[layer+1], layer_indices[layer+2]) # postsynaptic
	#	 weights_layer = W[layer_indices[layer+1]:layer_indices[layer+2], layer_indices[layer]:layer_indices[layer+1]] # weights between pre- and post-synaptic layers
	#	 weights_layer_dense = weights_layer.todense()
	#	 
	#	 net.linears[layer].weight = torch.nn.Parameter(torch.Tensor(weights_layer_dense)) # convert to tensor & update weights
	return net

def uniform_weights_to_tensor(net, low_bound=-0.05, high_bound=0.05, add_shift=0, mul_shift=1, frac_excitatory=0, device=device):
	for layer in net.linears:
		# get layer size
		layer_size = layer.weight.shape
		# pick weights from uniform distribution
		weights = (torch.empty(layer_size, device=device).uniform_(low_bound, high_bound) + add_shift) * mul_shift
		# choose excitatory links (outgoing)
		exc_indc = torch.randperm(layer_size[1])[:int(layer_size[1]*frac_excitatory)]
		# set excitatory neurons
		weights[:,exc_indc] *= -1
		# transfer weights to network instance
		layer.weight = torch.nn.Parameter(weights)
	return net

def convert_index_to_nested_struc(indices, n_l):
	'''
	function that convertes an given index in standard representation
	to the corresponding order structure which is given by the output 
	of the used ML-algorithm. the structure is: [i][0][j] where i is 
	the layer and j is the intra-layer index.
	'''
	
	# get indices to corresponding layers
	# entries correspond to input/output of corresponding layer i, where i is index of this array
	layer_indices = torch.cat([torch.tensor([0], device=device), torch.cumsum(torch.tensor(n_l, device=device), dim=0)])
	
	# in which layer does index live?
	home_layer = torch.tensor([next(layer-1 for layer, lower_index_border in enumerate(layer_indices) if index<lower_index_border) for index in indices], device=device)
	lower_index_border = layer_indices[home_layer]
	
	# corresponding intra-layer index
	home_layer_idx = indices - lower_index_border
	
	return home_layer, home_layer_idx

@njit
def find_homelayer(indices, layer_indices):
	'''
	This function finds the layer a given index is living in.
	'''

	# allocate array holding the homelayers corresponding to the indices
	homelayer = np.empty(len(indices), dtype=np.int64)

	# iterate all indices
	for i, index in enumerate(indices):

		# compare index to index borders of layers
		for layer, lower_index_border in enumerate(layer_indices):

			# save and break if index is smaller than border of next layer
			if index < lower_index_border:
				homelayer[i] = layer-1
				break
		else:
			# only raise error in case condition was never met
			raise RuntimeError

	return homelayer

@njit
def convert_index_to_nested_struc_numpy(indices, n_l):
	'''
	function that convertes an given index in standard representation
	to the corresponding order structure which is given by the output
	of the used ML-algorithm. the structure is: [i][0][j] where i is
	the layer and j is the intra-layer index.
	'''

	# get indices to corresponding layers
	# entries correspond to input/output of corresponding layer i, where i is index of this array
	layer_indices = np.concatenate((np.array([0]), np.cumsum(n_l)))

	# in which layer does index live?
	#home_layer = np.array([next(layer - 1 for layer, lower_index_border in enumerate(layer_indices) if index < lower_index_border) for index in indices])
	home_layer = find_homelayer(indices, layer_indices)
	lower_index_border = layer_indices[home_layer]

	# corresponding intra-layer index
	home_layer_idx = indices - lower_index_border

	return home_layer, home_layer_idx

def convert_nested_struc_to_index(layers, layerwise_indices, n_l):
	'''
	function that converts given nested indices in to the corresponding index
	in standard representation. this is the reverse function to 
	convert_index_to_nested_struc(index, n_l).
	'''
	
	# get indices to corresponding layers
	# entries correspond to input/output of corresponding layer i, where i is index of this array
	layer_indices = torch.cat([torch.tensor([0], device=device), torch.cumsum(torch.tensor(n_l, device=device), dim=0)])
	
	indices = layer_indices[layers] + layerwise_indices
	
	return indices

@njit
def convert_nested_struc_to_index_numpy(layers, layerwise_indices, n_l):
	'''
	function that converts given nested indices in to the corresponding index
	in standard representation. this is the reverse function to
	convert_index_to_nested_struc(index, n_l).
	'''

	# get indices to corresponding layers
	# entries correspond to input/output of corresponding layer i, where i is index of this array
	layer_indices = np.concatenate((np.array([0]), np.cumsum(n_l)))

	indices = layer_indices[layers] + layerwise_indices

	return indices

# pure pytorch implementation of bornholdt rewiring rule
def self_organize_couplings_torch(activations, weights, activity_threshold, nr_neurons, add_factor, n_l):
	'''
	this function is an updated version of self_organize_couplings() from binary_neuron_network.py.
	it inputs the activations and weights as they are output from net.trace and net.linears[:].weight
	respectively. the used ML algorithm is plain gradient descent, the weights are calculated 
	using the gradients of the weights w.r.t. to the loss.
	'''
	
	# only use for plain gradient descent, no batching
	if not batchsize == 1:
		raise ValueError('batch size is set != 1. Rewiring rule not defined. Aborting.')

	N = sum(n_l)
	
	#t11 = time()

	# choose random indices
	pool = torch.arange(n_l[0], N, device=device)  # pool of indices to draw from, dont include input layer
	random_indices = pool[torch.randperm(len(pool))[:nr_neurons]]
	
	#print('find random indices: {}'.format(time()-t11))

	#t12 = time()

	# convert indices to nested structure
	random_layers, random_il_indices = convert_index_to_nested_struc(random_indices, n_l)
	random_presyn_layers = random_layers - 1

	# get activities and coupling strength (have to use for loop?)
	random_il_links = torch.empty(nr_neurons, dtype=int, device=device)
	avrg_activities = torch.empty(nr_neurons, device=device)
	old_coupling_strengths = torch.empty(nr_neurons, device=device)
	for i, (random_layer, random_il_index) in enumerate(zip(random_layers, random_il_indices)):
		random_il_links[i] = torch.randperm(n_l[random_presyn_layers[i]], dtype=int)[0]
		avrg_activities[i] = torch.abs(activations[random_layer][0][random_il_index])
		old_coupling_strengths[i] = weights[random_layer - 1][random_il_index, random_il_links[i]]

	# determine factor sign
	add_factor_sign = torch.ones(nr_neurons, device=device) * add_factor
	# negative if node is active
	add_factor_sign[avrg_activities > activity_threshold] *= -1
	
	# get sign of old couplings
	coupling_signs = torch.sign(old_coupling_strengths)
	# compute new coupling strengths
	new_coupling_strengths = old_coupling_strengths + coupling_signs*add_factor_sign

	#print('find random links and calc new couplings: {}'.format(time()-t12))

	# convert nested structure to indices
	random_links = convert_nested_struc_to_index(random_presyn_layers, random_il_links, n_l)

	is_active = [not bool_val for bool_val in avrg_activities < activity_threshold]
	
	#return P_copy, torch.cat((random_indices, random_link, new_coupling_strength, avrg_activity, [not bool_val for bool_val in avrg_activity < activity_threshold]))
	return random_indices, random_links, new_coupling_strengths, avrg_activities, is_active

@njit
#@jit(debug=True, nopython=True, parallel=False, cache=True)
def get_new_couplings(random_layers, random_il_indices, random_presyn_layers, nr_neurons, n_l, add_factor, activity_threshold, activations, weights, rand_rewiring=False):
	'''
	This function loads the weights and activations, calculates the average activities and outputs
	the new coupling strengths with the chosen links and corresponding average activities.
	Performance is increased with numba.
	'''

	# get activities and coupling strength
	current_batchsize = len(activations[0])
	random_il_links = np.empty(nr_neurons, dtype=np.int64)
	avrg_activities = np.empty((nr_neurons, current_batchsize), dtype=np.float64)
	old_coupling_strengths = np.empty(nr_neurons, dtype=np.float64)
	add_factor_sign = np.ones((nr_neurons, current_batchsize), dtype=np.float64) * add_factor
	#for i, (random_layer, random_il_index) in enumerate(zip(random_layers, random_il_indices)):
	for i in range(len(random_layers)):
		# get random link
		#il_pool = np.empty(n_l[random_presyn_layers[i]], dtype=uint) # get empty array
		#for j in prange(il_pool.size):
		#	 il_pool[j] = j # fill empty array (np.arange not supportet with dtype argument)
		il_pool = np.arange(n_l[random_presyn_layers[i]], dtype=np.int64) # dtype=uint
		random_il_links[i] = np.random.choice(il_pool, (1,))[0]
		
		# get old coupling strength
		old_coupling_strengths[i] = weights[random_layers[i] - 1][random_il_indices[i], random_il_links[i]]
		
		# save activity of each sample in batch
		for sample, layer_activity in enumerate(activations[random_layers[i]]):
			avrg_activities[i,sample] = np.abs(layer_activity[random_il_indices[i]])
	
	# determine factor sign
	if rand_rewiring:
		is_active = np.random.choice(np.array([True, False]), avrg_activities.shape)
	else:
		is_active = avrg_activities > activity_threshold

	for i in range(nr_neurons):
		for j in range(current_batchsize):
			if is_active[i,j]:
				add_factor_sign[i,j] *= -1.
	#add_factor_sign[is_active] *= -1.0

	# compute new coupling strengths
	coupling_sign = np.sign(old_coupling_strengths)
	coupling_sign_batch = coupling_sign.repeat(current_batchsize).reshape(-1,current_batchsize) # axis=0 neuron, axis=1 sample
	# compute for whole batch
	old_cs_batch = old_coupling_strengths.repeat(current_batchsize).reshape(-1,current_batchsize) # axis=0 neuron, axis=1 sample
	new_cs_batch = old_cs_batch + coupling_sign_batch*add_factor_sign
	#new_cs_batch = old_cs_batch + add_factor_sign
	
	# average new coupling strength
	#new_coupling_strengths = np.mean(new_cs_batch, axis=0) # average over batches
	new_coupling_strengths = np.full_like(old_coupling_strengths,0)
	for i, coupling_batch in enumerate(new_cs_batch):
		new_coupling_strengths[i] = coupling_batch.mean()
	conditions = 0
	return old_coupling_strengths, new_coupling_strengths, random_il_links, avrg_activities, is_active, conditions

# rewiring process considering presynaptic activations
#@jit(debug=True, nopython=True, parallel=False, cache=True)
@njit
def get_new_couplingsNEW(random_layers, random_il_indices, random_presyn_layers, nr_neurons, n_l, add_factor, activity_threshold, activations, weights):
	'''
	This function loads the weights and activations, calculates the average activities and outputs
	the new coupling strengths with the chosen links and corresponding average activities.
	Performance is increased with numba.
	'''

	# get activities and coupling strength
	current_batchsize = len(activations[0])
	random_il_links = np.empty(nr_neurons, dtype=np.int64)
	batch_activation = np.empty((nr_neurons, current_batchsize), dtype=np.float64)
	old_coupling_strengths = np.empty(nr_neurons, dtype=np.float64)
	add_factor_sign = np.ones((nr_neurons, current_batchsize), dtype=np.float64) * add_factor
	presyn_activation = np.empty((nr_neurons, current_batchsize), dtype=np.float64)
	presyn_activation_sign = np.empty((nr_neurons, current_batchsize), dtype=np.float64)
	#for i, (random_layer, random_il_index) in enumerate(zip(random_layers, random_il_indices)):
	for i in range(len(random_layers)):
		# get random link
		#il_pool = np.empty(n_l[random_presyn_layers[i]], dtype=uint) # get empty array
		#for j in prange(il_pool.size):
		#	 il_pool[j] = j # fill empty array (np.arange not supportet with dtype argument)
		il_pool = np.arange(n_l[random_presyn_layers[i]], dtype=np.int64) # dtype=uint
		random_il_links[i] = np.random.choice(il_pool, (1,))[0]
		
		# get old coupling strength
		old_coupling_strengths[i] = weights[random_layers[i] - 1][random_il_indices[i], random_il_links[i]]
		
		# save activity of each sample in batch
		for sample, layer_activity in enumerate(activations[random_layers[i]]):
			batch_activation[i,sample] = layer_activity[random_il_indices[i]]
		
		# get sign of presyn neuron
		for sample, presyn_layer_activity in enumerate(activations[random_presyn_layers[i]]):
			#presyn_activation[i,sample] = presyn_layer_activity[random_presyn_layers[i]]
			#presyn_activation_sign[i,sample] = np.sign(presyn_layer_activity[random_presyn_layers[i]])
			presyn_activation[i,sample] = presyn_layer_activity[random_il_links[i]]
			presyn_activation_sign[i,sample] = np.sign(presyn_activation[i,sample])
	
	#determine whether activations should be de- or increased
	# -T < a_i < 0
	condition1 = np.logical_and(-activity_threshold<batch_activation, batch_activation<0)
	# +T < a_i
	condition2 = activity_threshold<batch_activation
	# determine where to de- or increase activations
	change_act = np.logical_or(condition1, condition2)
	# re-model the matrix such that it can be used as multiplier to compute new couplings
	change_act = -1*change_act+~change_act
	# save for debugging
	conditions = (condition1, condition2)

	# # get signs of postsynaptic activagtions
	# postsyn_activation_sign = np.sign(batch_activation)
	
	# get signs of old coupling strengths
	coupling_sign = np.sign(old_coupling_strengths)
	#coupling_sign_batch = coupling_sign.repeat(current_batchsize).reshape(-1,current_batchsize) # axis=0 neuron, axis=1 sample

	# if presynaptic activations are 0, define sign of 1 to enable weight update
	#presyn_activation_sign[presyn_activation_sign==0] = 1
	for xiter in range(nr_neurons):
		for yiter in range(current_batchsize):
			if presyn_activation_sign[xiter, yiter] == 0:
				presyn_activation_sign[xiter, yiter] = 1
	
	# compute new coupling strengths
	old_cs_batch = old_coupling_strengths.repeat(current_batchsize).reshape(-1,current_batchsize) # axis=0 neuron, axis=1 sample
	new_cs_batch = old_cs_batch + presyn_activation_sign*change_act*add_factor
	
	# # check if rewiring is right
	# check = np.full_like(np.logical_or(condition1, condition2),0)
	# for i, batch in enumerate(np.logical_or(condition1, condition2)):
	#	  for j, decrease in enumerate(batch):
	#		  pre_rewiring = old_cs_batch[i,j]*presyn_activation[i,j]
	#		  post_rewiring = new_cs_batch[i,j]*presyn_activation[i,j]
	#		  if decrease:
	#			  check[i,j] = pre_rewiring>post_rewiring
	#		  else:
	#			  check[i,j] = pre_rewiring<post_rewiring
	#import pdb; pdb.set_trace()
	# average new coupling strength
	#new_coupling_strengths = np.mean(new_cs_batch, axis=0) # average over batches
	new_coupling_strengths = np.full_like(old_coupling_strengths,0)
	for i, coupling_batch in enumerate(new_cs_batch):
		new_coupling_strengths[i] = coupling_batch.mean()
	
	return old_coupling_strengths, new_coupling_strengths, random_il_links, batch_activation, change_act, conditions

#saved rewiring without random manipulation of weights
@njit
#@jit(debug=True, nopython=True, parallel=False, cache=True)
def get_new_couplingsNEW2(random_layers, random_il_indices, random_presyn_layers, nr_neurons, n_l, add_factor, activity_threshold, activations, weights):
	'''
	This function loads the weights and activations, calculates the average activities and outputs
	the new coupling strengths with the chosen links and corresponding average activities.
	Performance is increased with numba.
	'''

	# get activities and coupling strength
	current_batchsize = len(activations[0])
	random_il_links = np.empty(nr_neurons, dtype=np.int64)
	avrg_activities = np.empty((nr_neurons, current_batchsize), dtype=np.float64)
	old_coupling_strengths = np.empty(nr_neurons, dtype=np.float64)
	add_factor_sign = np.ones((nr_neurons, current_batchsize), dtype=np.float64) * add_factor
	#for i, (random_layer, random_il_index) in enumerate(zip(random_layers, random_il_indices)):
	for i in range(len(random_layers)):
		# get random link
		#il_pool = np.empty(n_l[random_presyn_layers[i]], dtype=uint) # get empty array
		#for j in prange(il_pool.size):
		#	 il_pool[j] = j # fill empty array (np.arange not supportet with dtype argument)
		il_pool = np.arange(n_l[random_presyn_layers[i]], dtype=np.int64) # dtype=uint
		random_il_links[i] = np.random.choice(il_pool, (1,))[0]
		
		# get old coupling strength
		old_coupling_strengths[i] = weights[random_layers[i] - 1][random_il_indices[i], random_il_links[i]]
		
		# save activity of each sample in batch
		for sample, layer_activity in enumerate(activations[random_layers[i]]):
			avrg_activities[i,sample] = np.abs(layer_activity[random_il_indices[i]])
	
	# determine factor sign
	is_active = avrg_activities > activity_threshold
	for i in range(nr_neurons):
		for j in range(current_batchsize):
			if is_active[i,j]:
				add_factor_sign[i,j] *= -1.
	#add_factor_sign[is_active] *= -1.0

	# compute new coupling strengths
	coupling_sign = np.sign(old_coupling_strengths)
	coupling_sign_batch = coupling_sign.repeat(current_batchsize).reshape(-1,current_batchsize) # axis=0 neuron, axis=1 sample
	# compute for whole batch
	old_cs_batch = old_coupling_strengths.repeat(current_batchsize).reshape(-1,current_batchsize) # axis=0 neuron, axis=1 sample
	new_cs_batch = old_cs_batch + coupling_sign_batch*add_factor_sign
	#new_cs_batch = old_cs_batch + add_factor_sign
	
	# average new coupling strength
	#new_coupling_strengths = np.mean(new_cs_batch, axis=0) # average over batches
	new_coupling_strengths = np.full_like(old_coupling_strengths,0)
	for i, coupling_batch in enumerate(new_cs_batch):
		new_coupling_strengths[i] = coupling_batch.mean()
	conditions = 0
	return old_coupling_strengths, new_coupling_strengths, random_il_links, avrg_activities, is_active, conditions

@njit
def get_new_couplingsNEW3(random_layers, random_il_indices, random_presyn_layers, nr_neurons, n_l, add_factor, activity_threshold, activations, weights, rand_rewiring=False):
	'''
	This function loads the weights and activations, calculates the average activities and outputs
	the new coupling strengths with the chosen links and corresponding average activities.
	Performance is increased with numba.
	'''

	# get activities and coupling strength
	current_batchsize = len(activations[0])
	random_il_links = np.empty(nr_neurons, dtype=np.int64)
	avrg_activities = np.empty((nr_neurons, current_batchsize), dtype=np.float64)
	old_coupling_strengths = np.empty(nr_neurons, dtype=np.float64)
	add_factor_sign = np.ones((nr_neurons, current_batchsize), dtype=np.float64) * add_factor
	mul_factor = np.ones((nr_neurons, current_batchsize), dtype=np.float64) * 2.0
	#for i, (random_layer, random_il_index) in enumerate(zip(random_layers, random_il_indices)):
	for i in range(len(random_layers)):
		# get random link
		#il_pool = np.empty(n_l[random_presyn_layers[i]], dtype=uint) # get empty array
		#for j in prange(il_pool.size):
		#	 il_pool[j] = j # fill empty array (np.arange not supportet with dtype argument)
		il_pool = np.arange(n_l[random_presyn_layers[i]], dtype=np.int64) # dtype=uint
		random_il_links[i] = np.random.choice(il_pool, (1,))[0]
		
		# get old coupling strength
		old_coupling_strengths[i] = weights[random_layers[i] - 1][random_il_indices[i], random_il_links[i]]
		
		# save activity of each sample in batch
		for sample, layer_activity in enumerate(activations[random_layers[i]]):
			avrg_activities[i,sample] = np.abs(layer_activity[random_il_indices[i]])
	
	# determine factor sign
	if rand_rewiring:
		is_active = np.random.choice(np.array([True, False]), avrg_activities.shape)
	else:
		is_active = avrg_activities > activity_threshold # ~np.isclose(avrg_activities,0.) 

	for i in range(nr_neurons):
		for j in range(current_batchsize):
			# negative constant if neuron > 0
			if is_active[i,j]:
				# add_factor_sign[i,j] *= -1.
				mul_factor[i,j] = 0.5
			# random constant if activity close to zero
			# if np.isclose(avrg_activities[i,j], 0.):
			#	  add_factor_sign[i,j] *= np.random.choice(np.array([-1, 1]))
	#add_factor_sign[is_active] *= -1.0

	# compute new coupling strengths
	# coupling_sign = np.sign(old_coupling_strengths)
	# coupling_sign_batch = coupling_sign.repeat(current_batchsize).reshape(-1,current_batchsize) # axis=0 neuron, axis=1 sample
	# compute for whole batch
	old_cs_batch = old_coupling_strengths.repeat(current_batchsize).reshape(-1,current_batchsize) # axis=0 neuron, axis=1 sample
	new_cs_batch = old_cs_batch * mul_factor
	# new_cs_batch = old_cs_batch + coupling_sign_batch * add_factor_sign
	#new_cs_batch = old_cs_batch + add_factor_sign
	
	# average new coupling strength
	#new_coupling_strengths = np.mean(new_cs_batch, axis=0) # average over batches
	new_coupling_strengths = np.full_like(old_coupling_strengths,0)
	for i, coupling_batch in enumerate(new_cs_batch):
		new_coupling_strengths[i] = coupling_batch.mean()
	conditions = 0
	return old_coupling_strengths, new_coupling_strengths, random_il_links, avrg_activities, is_active, conditions
@njit
#@jit(debug=True, nopython=True, parallel=False, cache=True)
def self_organize_couplings_numpy(activations, weights, activity_threshold, nr_neurons, add_factor, n_l, random_rewiring):
	'''
	inputs the activations and weights as they are output from net.trace and net.linears[:].weight
	respectively.
	'''

	N = np.sum(n_l)

	# choose random indices
	# rng = np.random.default_rng() # initialize generator object
	# random_indices = rng.choice(N-n_l[0], nr_neurons) # generate uniform random sample from [0,N-n_l[0]] of size nr_neurons
	# random_indices = random_indices + n_l[0] # leave out first layer -> map to right indices
	pool = np.arange(n_l[0], N-n_l[-1], dtype=np.int64) # neglect input and output layer
	random_indices =  np.random.choice(pool, nr_neurons, replace=False) # select indices from pool

	# convert indices to nested structure
	random_layers, random_il_indices = convert_index_to_nested_struc_numpy(random_indices, n_l)
	random_presyn_layers = random_layers - 1

	# get activities, coupling strengths and random links (optimized with numba)
	old_coupling_strengths, new_coupling_strengths, random_il_links, avrg_activities, is_active, conditions = get_new_couplingsNEW3(
		random_layers,
		random_il_indices,
		random_presyn_layers,
		nr_neurons,
		n_l,
		add_factor,
		activity_threshold,
		List(activations),
		List(weights),
		rand_rewiring=random_rewiring
	)

	# convert nested structure to indices
	random_links = convert_nested_struc_to_index_numpy(random_presyn_layers, random_il_links, n_l)
	
	# return P_copy, torch.cat((random_indices, random_link, new_coupling_strength, avrg_activity, [not bool_val for bool_val in avrg_activity < activity_threshold]))
	return random_indices, random_links, old_coupling_strengths, new_coupling_strengths, avrg_activities, is_active, conditions

#@njit(debug=True)
def loop_weight_update(net, pre_home, il_post_indices, il_pre_indices, new_couplings):
	for layer, post_idx, pre_idx, coupling in zip(pre_home, il_post_indices, il_pre_indices, new_couplings):
		net.linears[layer].weight[post_idx, pre_idx] = coupling

	return net

@njit
def convert_to_flat_weights(pre_home, il_post_indices, il_pre_indices, n_l):

	# length nested weight generator
	dim = len(n_l)-1

	# initialize array holding linear indices
	flat_indices_layers = np.empty(dim, dtype=np.int64)

	# iterate to calulate number of weights (flat indices) in each layer
	for i in range(dim):
		flat_indices_layers[i] = n_l[i]*n_l[i+1]

	# calculate overall flat indices
	flat_indices = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(flat_indices_layers)))

	# calculate indices by equation: i = k + j * s1 where i = flat index, j,k = cartesian indices, s1 = len of dim1
	flat_il_indices = il_pre_indices + il_post_indices * n_l[pre_home]

	flat_weight_indices = flat_indices[pre_home] + flat_il_indices

	return flat_weight_indices

def update_weights(net, post_indices, pre_indices, new_couplings, n_l, device):
	'''
	this function inputs a instantiated network, random indices (postsynaptic neuron),
	random links (presynaptic neuron) and the new coupling strength between them.
	returns the network with accordingly customized weights.
	'''

	#t1 = time()

	#N = np.sum(n_l)

	post_home, il_post_indices = convert_index_to_nested_struc_numpy(post_indices, n_l)
	pre_home, il_pre_indices = convert_index_to_nested_struc_numpy(pre_indices, n_l)

	#print('convert indices: {}s'.format(time()-t1))

	# # sanity check
	# if not np.all(pre_home == post_home-1): raise Exception(
	#	  '''Post- and pre-synaptic neuron arent living in consecutive layers.
	#	  something went wrong. Aborting'''
	#	  )

	#before=net.linears[pre_home[0]].weight[il_post_indices[0], il_pre_indices[0]]
	#print('weight before rewiring: {}'.format(before))
	#t2 = time()

	# transfer weight to network instance

	with torch.no_grad(): # change is not reflected in gradients
		#net = loop_weight_update(net, pre_home, il_post_indices, il_pre_indices, new_couplings)
		weights = torch.nn.utils.parameters_to_vector(net.parameters())
		weights_old = copy.deepcopy(weights)
		flat_indices = convert_to_flat_weights(pre_home, il_post_indices, il_pre_indices, n_l)
		weights[flat_indices] = torch.tensor(new_couplings, dtype=torch.float32, device=device)
		#print('flat weights same? {}'.format(torch.all(weights_old==weights)))
		torch.nn.utils.vector_to_parameters(weights, net.parameters())

	#print('set weights: {}s'.format(time()-t2))
	#after = net.linears[pre_home[0]].weight[il_post_indices[0], il_pre_indices[0]]
	#print('weight after rewiring: {}. same? {}'.format(after, before==after))

	return net

def array_from_ragged_list(ragged_list):
	"""
	This function converts an ragged nested list to an numpy array.
	It takes the length of the biggest entry list, and fills missing
	entries with nans. This way readability is preserved when saved as *.csv
	"""

	length_list = len(ragged_list)
	max_length_entry = len(max(ragged_list, key=len))

	a = np.full((max_length_entry, length_list), np.nan)

	for i, l in enumerate(ragged_list):
		length_row = len(l)
		a[0:length_row, i] = l

	return a

def floor_to_decimal(number, decimal_place):
	return int(math.floor(number / decimal_place)) * decimal_place
