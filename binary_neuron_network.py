import numpy as np
import shutil, os, math, sys, datetime, pickle, copy, statistics
from scipy.sparse import csc_matrix
import scipy.sparse
import reconstruct_couplings_binary_neuron_network
from numba.typed import List

start_time = datetime.datetime.now()
print(f"\nStarting {sys.argv[0]} at {start_time}")

from PARAM_binary_neuron_network import * # load simulation parameters
from NN_set_up import * # load functions for torch implementation

### command line parameters ###

#S = int(sys.argv[2]) # number of initialy active links given by command line argument
m = float(sys.argv[1]) # parameter controlling neural excitability

S_is_string = False
try:
    S = int(sys.argv[2]) # number of initialy active links given by command line argument
except ValueError:
    S = sys.argv[2]
    S_is_string = True
    print('\nS = ARGV[2] is not a number. Randomly choose S\n')

###############################

# get device (cpu/gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # check available device
print('torch_impl set to {}\ntorch.device = {}'.format(torch_impl, device))

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

def round2precision(val, precision: int = 0, which: str = ''):
    assert precision >= 0
    val *= 10 ** precision
    round_callback = round
    if which.lower() == 'up':
        round_callback = math.ceil
    if which.lower() == 'down':
        round_callback = math.floor
    return '{1:.{0}f}'.format(precision, round_callback(val) / 10 ** precision)

def percolate_shew09(X,P,zeta,t,feedforward): # perculation function shew et al. 2009
    #J = [True if X[i,t] == True else False for i in range(N)] # group spiking neurons
    #prod_term = 1 - P[:,J]*a[:,J]
    if feedforward == False:
        prod_term = 1 - P*X[:,t]
    else:
        prod_term = 1 - P.multiply(X[:,t]).toarray()
    
    x = 1 - np.prod(prod_term,1) - zeta[:,t]
    X[:,t+1] = np.heaviside(x,0)
    return X[:,t+1]#, np.prod(prod_term,1)

def cases_meisel20(x, p_ne):
    if x < 0:
        return 0
    elif x < 1:
        return x*p_ne
    else:
        return p_ne

def percolate_meisel20(X,W,p_ne,t,feedforward): # X state matrix, W coupling weights, p_ne parameter controlling neural excitability, t time, N number of neurons
    if feedforward == False:
        P = np.sum(np.multiply(W,np.transpose(X[:,t])), axis=1) # sum of inputs
    else:
        P = W.multiply(X[:,t]).sum(axis=1)
    
    fire_probability = [cases_meisel20(float(x),p_ne) for x in P] #map(lambda x : cases_meisel20(x,p_ne), P)
    X[:,t+1] = np.random.uniform(0,1,len(P)) < fire_probability
    return X[:,t+1]

def construct_feedforward_adjacency(n_l):
    A = np.zeros((sum(n_l),sum(n_l)),bool)

    idx = np.arange(sum(n_l))
    n_l = np.concatenate([[0],n_l])
    
    nodes_past = 0
    for i in np.arange(len(n_l)-2): # iterate each layer (not the last one)
        #print(f'iteration {i}')
        nodes_past += n_l[i]
        nodes_current = n_l[i+1]
        nodes_next = n_l[i+2]
        idx_current = idx[nodes_past:nodes_past+nodes_current]
        idx_next = idx[nodes_past+nodes_current:nodes_past+nodes_current+nodes_next]
        #print(f'nodes_past = {nodes_past}; node_current = {nodes_current}; nodes_next = {nodes_next}')
        #print(f'idx_current = {idx_current}; idx_next = {idx_next}')

        for j in idx_current:
            for k in idx_next:
                A[k,j] = True

    return A

def construct_coupling_strengths(A, mean, stdev): # A = coupling matrix, resulting matrix has mean = mean and standard deviation = stdev
    check_binary = np.array_equal(A, A.astype(bool))
    if check_binary == False:
        return -1
    
    N = len(A)
    nlinks = np.sum(A)
    nnolinks = N**2 - nlinks
    
    lower_bound = N**2 * mean / nlinks - np.sqrt(3) * stdev #(N**3*mean - np.sqrt(3)*nlinks)/(nlinks*N)
    upper_bound = N**2 * mean / nlinks + np.sqrt(3) * stdev #(N**3*mean + np.sqrt(3)*nlinks)/(nlinks*N) 
    P_values_full_connect = np.random.uniform(lower_bound,upper_bound,(N,N)) # no links
    P_values_disconnect = np.multiply(P_values_full_connect,A) #np.concatenate([P_values_no_zeros,np.zeros(nnolinks,bool)]) # add links to array
    #P = np.random.choice(P_values_w_zeros, size=(N,N), replace=False)
    return P_values_disconnect

def construct_uniform_feedforward_couplings_layerwise(n_l, mean, stdev, layer): #  resulting matrix has mean = mean and standard deviation = stdev, layer = present layer at time t
    
    # only use for feedforward nets
    check_feedforward = len(n_l) > 1
    if check_feedforward == False:
        return -1
    
    # number of units
    N = np.sum(n_l)
    
    # number of links in network
    nlinks = 0
    for i in range(len(n_l)-1):
        nlinks += n_l[i] * n_l[i+1]
    
    # boundaries for random coupling strengths
    lower_bound = N**2 * mean / nlinks - np.sqrt(3) * stdev
    upper_bound = N**2 * mean / nlinks + np.sqrt(3) * stdev
    
    # layout of coupling matrix layer t -> t+1
    # this matrix maps the connection from unit j (at time t and layer n) to unit i (at time t+1 and layer n+1)
    # and therefore differs from a normal adjacency matrix
    max_units_per_layer = np.max(n_l)
    P_layer = np.zeros((max_units_per_layer, max_units_per_layer), float)
    P_values = np.random.uniform(lower_bound, upper_bound, (n_l[layer+1], n_l[layer]))
    for x in np.ndenumerate(P_values):
        P_layer[x[0]] = x[1]
    
    return P_layer

def construct_uniform_feedforward_couplings_sparse(n_l, mean, stdev, path, sub_path_m, sub_path_S, num_path, iter):
    
    # only use for feedforward nets
    check_feedforward = len(n_l) > 1
    if check_feedforward == False:
        return -1
    
    # number of units
    N = np.sum(n_l)
    
    # number of links in network
    nlinks = np.sum([n_l[i] * n_l[i+1] for i in range(len(n_l)-1)])
    
    # boundaries for random coupling strengths
    lower_bound = N**2 * mean / nlinks - np.sqrt(3) * stdev
    upper_bound = N**2 * mean / nlinks + np.sqrt(3) * stdev
    
    # construct matrix in coordinate format

    col = np.full(nlinks, np.nan)
    row = np.full(nlinks, np.nan)
    unit_nr_per_layer = np.concatenate([[0],np.cumsum(n_l)])
    
    # implemented as written for loop (slow)
    
#   nr_of_link = 0
#   for layer in range(len(unit_nr_per_layer)-2): # iterate over all layers
#       for pre_unit in range(unit_nr_per_layer[layer], unit_nr_per_layer[layer+1]): # the layer where the signal is coming from
#           for post_unit in range(unit_nr_per_layer[layer+1], unit_nr_per_layer[layer+2]): # the layer where the signal is going to
#               print('pre_unit = {}'.format(pre_unit))
#               print('post_unit = {}'.format(post_unit))
#               col[nr_of_link] = pre_unit
#               row[nr_of_link] = post_unit
#               
#               nr_of_link += 1
#   
#   layer = [i for i in range(len(unit_nr_per_layer)-2)]
#   pre_unit = [i for l in layer for i in range(unit_nr_per_layer[l], unit_nr_per_layer[l+1])]
#   post_unit = [i for l in layer for i in range(unit_nr_per_layer[l+1], unit_nr_per_layer[l+2])]
#   
    # implemented as list comprehension (fast?)
    
    idx = [[pre, post] for l in range(len(unit_nr_per_layer)-2) for pre in range(unit_nr_per_layer[l], unit_nr_per_layer[l+1]) for post in range(unit_nr_per_layer[l+1], unit_nr_per_layer[l+2])]
    idx = np.array(idx)
    
    statesave_RNG('W', iter, path+sub_path_m+sub_path_S+num_path) # save RNG seed
    data = np.random.uniform(lower_bound, upper_bound, nlinks)
    csc = csc_matrix((data, (idx[:,1], idx[:,0])), shape = (N,N))
    
    return csc
    
def self_organize_couplings(X, P, T1, T2, activity_threshold, nr_neurons, add_factor): # X state matrix, P coupling strength matrix, m control parameter, T1 - T2 start - endpoint integration average activity, nr_neurons how many neurons get updated
    
    N = len(X[:,0])
    P_copy = copy.deepcopy(P) # copy so original P wont change
    
    random_indices = np.zeros(nr_neurons, int)
    adjacent_units = np.zeros(nr_neurons, object)
    unit_has_incoming_link = np.full(nr_neurons, False)
    pool = np.arange(N) # pool of indices to draw from
    count_iterations = 0 # >1000 --> error
    for i in range(nr_neurons):
        while unit_has_incoming_links[i] == False:
            
            # check how many times already iterated
            if count_iterations > 1000:
                RuntimeError('self_organize_couplings() could not find any indices with incoming links. Abort.')
            
            # choose random index and delete from pool of indices
            random_indices[i] = np.random.choice(pool, 1) # choose random index from whole network
            pool.remove(random_indices[i])
            
            if scipy.sparse.issparse(P):
                adjacent_units[i] = np.where(np.squeeze(P[random_indices[i],:].toarray(),axis=0) != 0)[0]
            else:
                adjacent_units[i] = np.where(P[random_indices[i],:] != 0)[0]
            if len(adjacent_units[i]) > 0:
                unit_has_incoming_links = True
            
            count_iterations += 1
    
    avrg_activity = np.zeros(nr_neurons)
    new_coupling_strength = np.zeros(nr_neurons)
    random_link = np.zeros(nr_neurons, int)
    for random_index in enumerate(random_indices):
        unit_activity = X[int(random_index[1]),:]
        avrg_activity[random_index[0]] = np.sum(np.abs(unit_activity[T1:T2]))/(T2-T1) # average activity (temporal)
        
        #if avrg_activity <= activity_threshold: # inactive units
    #   if avrg_activity[random_index[0]] < activity_threshold: # inactive units
    #       random_scaling = np.random.uniform(1, 2, 1)[0]      
    #   else: # active links
    #       random_scaling = np.random.uniform(0, 1, 1)[0]
    #   #print('len(adjacent_units[random_index[0]]) = {}, incoming = {}'.format(len(adjacent_units[random_index[0]]), unit_has_incoming_links))
    #   random_link[random_index[0]] = np.random.choice(adjacent_units[random_index[0]], 1)[0]  
    #   new_coupling_strength[random_index[0]] = random_scaling * P_copy[random_index[1],random_link[random_index[0]]]
        if avrg_activity[random_index[0]] <= activity_threshold: # inactive units
            add_factor_sign = add_factor
        else:
            add_factor_sign = -add_factor
        
        random_link[random_index[0]] = np.random.choice(adjacent_units[random_index[0]], 1)[0]
        new_coupling_strength[random_index[0]] = P_copy[random_index[1],random_link[random_index[0]]] + add_factor_sign
        
        P_copy[random_index[1],random_link[random_index[0]]] = new_coupling_strength[random_index[0]]
    
    return P_copy, np.concatenate([random_indices, random_link, new_coupling_strength, avrg_activity, [not bool_val for bool_val in avrg_activity < activity_threshold]])

# def convert_index_to_nested_struc(index, n_l):
#     '''
#     function that convertes an given index in standard representation
#     to the corresponding order structure which is given by the output 
#     of the used ML-algorithm. the structure is: [i][0][j] where i is 
#     the layer and j is the intra-layer index.
#     '''
    
#     # get indices to corresponding layers
#     # entries correspond to input/output of corresponding layer i, where i is index of this array
#     layer_indices = np.concatenate([[0], np.cumsum(n_l)])
    
#     # in which layer does index live?
#     home_layer = next(layer-1 for layer, lower_index_border in enumerate(layer_indices) if index<lower_index_border)
#     lower_index_border = layer_indices[home_layer]
    
#     # corresponding intra-layer index
#     home_layer_idx = index-lower_index_border
    
#     return home_layer, home_layer_idx

# def convert_nested_struc_to_index(layer, layerwise_idx, n_l):
#     '''
#     function that convertes given nested indices in to the corresponding index 
#     in standard representation. this is the reverse function to 
#     convert_index_to_nested_struc(index, n_l).
#     '''
    
#     # get indices to corresponding layers
#     # entries correspond to input/output of corresponding layer i, where i is index of this array
#     layer_indices = np.concatenate([[0], np.cumsum(n_l)])
    
#     index = layer_indices[layer] + layerwise_idx
    
#     return index

# # pure pytorch implementation of bornholdt rewiring rule
# def self_organize_couplings_torch(activations, weights, T1, T2, activity_threshold, nr_neurons, add_factor, n_l):
#     '''
#     this function is an updated version of self_organize_couplings() from binary_neuron_network.py.
#     it inputs the activations and weights as they are output from net.trace and net.linears[:].weight
#     respectively. the used ML algorithm is plain gradient descent, the weights are calculated 
#     using the gradients of the weights w.r.t. to the loss.
#     '''
    
#     # only use for plain gradient descent, no batching
#     if not batch_size == 1:
#         raise ValueError('batch size is set > 1. Rewiring rule not defined. Aborting.')

#     N = sum(n_l)
    
#     # # select random indices and check if they have incoming connections
#     # unit_has_incoming_links = torch.full(torch.Size([nr_neurons]), False)
#     # count_tries = 0 # max number of tries: 1000 --> error
#     # random_layer = 0
#     # while torch.all(unit_has_incoming_links) == False:
#     #     #random_indices = torch.randperm(N)[:nr_neurons] # draw indices
#     #     random_indices = np.random.choice(range(n_l[0], N), nr_neurons) # dont draw from input layer
#     #     adjacent_units = [[] for i in range(nr_neurons)]
        
#     #     for i in range(nr_neurons): # iterate all drawn indices
            
#     #         # get layerwise (intra-layer) indices
#     #         random_layer, random_il_index = convert_index_to_nested_struc(random_indices[i], n_l)
            
#     #         if random_layer == 0: # input layer has no weights
#     #             unit_has_incoming_links = torch.full(torch.Size([nr_neurons]), False)
#     #             count_tries += 1
#     #             break
                
#     #         else:
#     #             # fetch layer weights
#     #             w_layer = weights[random_layer-1] # w_layer doesnt include input layer
                
#     #             # get incoming links
#     #             adjacent_unit_nested = torch.nonzero((w_layer[random_il_index,:] != 0)).flatten()
#     #             adjacent_units[i] = convert_nested_struc_to_index(random_layer-1, adjacent_unit_nested, n_l)
                
#     #             #check if index has incomming links
#     #             if len(adjacent_units[i]) > 0:
#     #                 unit_has_incoming_links[i] = True
#     #             else:
#     #                 unit_has_incoming_links = torch.full(torch.Size([nr_neurons]), False)
#     #                 count_tries += 1
#     #                 break
                
#     #     if count_tries > 1000: raise RuntimeError('Rewiring Rule could not find units with incoming links. Abort.')
        
#     random_indices = torch.zeros([nr_neurons], dtype=int)
#     adjacent_units = [[] for i in range(nr_neurons)]
#     unit_has_incoming_link = torch.full([nr_neurons], False)
#     pool = torch.arange(n_l[0], N) # pool of indices to draw from, dont include input layer
#     count_iterations = 0 # >1000 --> error
#     for i in range(nr_neurons):
#         while unit_has_incoming_link[i] == False:
            
#             # check how many times already iterated
#             if count_iterations > 1000:
#                 RuntimeError(f'self_organize_couplings() could not find any more indices with incoming links. Found {torch.sum(unit_has_incoming_link)} from {nr_neurons}. Abort.')
            
#             # choose random index and delete from pool of indices
#             random_index = pool[torch.randperm(len(pool))[0]] # choose random index from pool
#             pool = pool[pool!=random_index] # remove index from pool
            
#             # get layerwise (intra-layer) indices
#             random_layer, random_il_index = convert_index_to_nested_struc(random_index, n_l)
            
#             # get layer weights
#             w_layer = weights[random_layer-1] # w_layer doesnt include input layer
            
#             # get incoming links
#             adjacent_unit_nested = torch.nonzero((w_layer[random_il_index,:])).flatten()
            
#             # check if there are incoming links (should be, since links are not removed like in bornholdt)
#             if len(adjacent_unit_nested)>0:
#                 adjacent_units[i] = convert_nested_struc_to_index(random_layer-1, adjacent_unit_nested, n_l)
#                 random_indices[i] = random_index
#                 unit_has_incoming_link[i] = True
            
#             count_iterations += 1
    
#     # calculate average activity, select random link, and set new coupling strength
#     avrg_activities = torch.zeros(nr_neurons)
#     new_coupling_strengths = torch.zeros(nr_neurons)
#     random_links = torch.zeros(nr_neurons, dtype=int)
#     for i, random_index in enumerate(random_indices):
#         random_layer, random_il_index = convert_index_to_nested_struc(random_index, n_l) # intra-layer index and corresponding layer
#         unit_activity = activations[random_layer][0][random_il_index]
#         avrg_activities[i] = torch.sum(torch.abs(unit_activity))/(T2-T1)
        
#         if avrg_activities[i] <= activity_threshold: # inactive units
#             add_factor_sign = add_factor
#         else:
#             add_factor_sign = -add_factor

#         # select a random link
#         #random_links[i] = np.random.choice(adjacent_units[i]) # shuffle adjacent units and pick first one
#         random_link_permute = torch.randperm(len(adjacent_units[i]))[0]
#         random_links[i] = adjacent_units[i][random_link_permute]
        
#         # compute new coupling strength
#         random_layer, random_il_link = convert_index_to_nested_struc(random_links[i], n_l)
#         old_coupling_strength = weights[random_layer][random_il_index, random_il_link]
#         new_coupling_strengths[i] = old_coupling_strength + add_factor_sign
    
#     is_active = [not bool_val for bool_val in avrg_activities < activity_threshold]

#     #return P_copy, torch.cat((random_indices, random_link, new_coupling_strength, avrg_activity, [not bool_val for bool_val in avrg_activity < activity_threshold]))
#     return random_indices, random_links, new_coupling_strengths, avrg_activities, is_active

# def update_weights(net, post_idx, pre_idx, new_coupling):
#     '''
#     this function inputs a instantiated network, random indices (postsynaptic neuron),
#     random links (presynaptic neuron) and the new coupling strength between them.
#     returns the network with accordingly customized weights.
#     '''
    
#     # construct array representing network structure (n_l) and node indices
#     net_struc = np.zeros(len(net.linears)+1, int) # include input layer and concatenated zero
#     for i, module in enumerate(net.linears):
#         net_struc[i] = module.in_features
#     net_struc[-1] = net.linears[-1].out_features
#     layer_indices = np.cumsum(net_struc)
    
#     # in which layers do they live?
#     #post_home = next(layer-1 for layer, lower_index_border in enumerate(layer_indices) if post_idx<lower_index_border)
#     #pre_home = next(layer-1 for layer, lower_index_border in enumerate(layer_indices) if pre_idx<lower_index_border)
    
#     post_home, il_post_idx = convert_index_to_nested_struc(post_idx, net_struc)
#     pre_home, il_pre_idx = convert_index_to_nested_struc(pre_idx, net_struc)
    
#     # sanity check
#     if not pre_home == post_home-1: raise Exception(
#         '''Post- and pre-synaptic neuron arent living in consecutive layers.
#         something went wrong. Aborting'''
#         )
    
#     # calculate the intralayer indices 
#     #il_post_idx = post_idx - layer_indices[post_home]
#     #il_pre_idx = pre_idx - layer_indices[pre_home]
    
#     # transfer weight to network instance
#     with torch.no_grad(): # change is not reflected in gradients
#         net.linears[pre_home].weight[il_post_idx, il_pre_idx] = new_coupling
        
#     return net

def statesave_RNG(name, iterator, path):
    seed = np.random.get_state()
    with open(f'{path}{name}_seed{iterator}.obj', 'wb') as f:
        pickle.dump(seed, f)
    
    # open with:
    # with open('state.obj', 'rb') as f:
    # np.random.set_state(load(f))

def rebuild_statematrix(x, tstep):
    N = x.shape[0]
    x_full = np.zeros((N,tstep))
    for i in range(N):
        for j in range(tstep):
            if j in range(x.shape[1]):
                x_full[i,j] = x[i,j]
            else:
                x_full[i,j] = 0
    return x_full

# def trace_to_statematrix(trace, n_l, tstep, ic):
#     '''
#     inputs list with activated nodes per layer
#     (see net1() class, net.trace property). outputs
#     matrix with node indices in axis=0 and timtesteps in axis=1
#     '''
#
#     N = sum(n_l) # how many nodes in total?
#     cumN = np.concatenate([[0],np.cumsum(n_l)]) # which indices in which layer?
#
#     trace_copy = copy.copy(trace)
#     #trace_copy.insert(0,ic) # include initial conditions
#     X = torch.zeros((N, tstep)) # construct state matrix
#     for layer, vec in enumerate(trace_copy): # fill with entries
#         layer_start = cumN[layer]
#         layer_end = cumN[layer+1]
#         X[layer_start:layer_end, layer] = vec.to('cpu')
#     return np.array(X.detach())

def cut_statematrix(X):
    for i in range(X.shape[1]):
        if sum(X[:,i]) == 0 and i < X.shape[1]-1:
            X = X[:,:i+1]
            break
    return X

def sigma_dynamic(x, skip_layers, n_l): # x state vector, n_l layer vector
    '''
    calculates the criticality measure sigma_dynamics, defined as 
    \sigma_{dyn} = 1/nlayers \sum_{j=0}^{nlayers} A_{j+1}/A_{j},
    where nlayers is the number of layers, and A_j is the activity in layer j.
    Also outputs the layer activites, and the activity ratios. x is the state
    vector (axis=0) for each timestep (axis=1). skip_layers is the number of
    layers to skip for sigma_dyn calculation. n_l layer structure.
    '''
    
    nlayers = len(n_l) # number of layers
    node_indices = np.cumsum(np.concatenate([[0],n_l])) # node indices in layers
    xlayer_activity = np.zeros(nlayers) # preallocate array for layer activities
    activity_ratio = np.zeros(nlayers-1)
    
    for layer, node_idx in enumerate(node_indices[:-1]):
        layer_nodes = [idx for idx in range(node_indices[layer], node_indices[layer+1])]
        xlayer_activity[layer] = sum(np.abs(x[layer_nodes,layer]))
        
        if layer > 0:
            if xlayer_activity[layer-1] == 0:
                activity_ratio[layer-1] = 0
            else:
                activity_ratio[layer-1] = xlayer_activity[layer]/xlayer_activity[layer-1]*n_l[layer-1]/n_l[layer]
    
    sigma_dynamic = sum(activity_ratio[skip_layers:])/(nlayers-skip_layers)
    
    return xlayer_activity, activity_ratio, sigma_dynamic


def iterate_percolation_process(S, tstep, iterations, n_l, activity_threshold):
    
    N = sum(n_l) # number of nodes
    l = len(n_l) # number of layers

    # pre allocate arrays
     
    #W = np.zeros((N,N))
    zeta = np.zeros((N,tstep))
    #idx = np.zeros(N, int)
    #ic_idx = np.zeros(S, int)
    mask = np.zeros(N, bool)
    
    if len(n_l) != 1:
        feedforward = True
        A = construct_feedforward_adjacency(n_l) # feedforward topology
    else:
        feedforward = False
        A = np.ones((N,N))
    
    if load_external_couplings:
        path = f"data.{model}.rewire_{local_rewiring}.N_{N}.l_{l}.tstep_{tstep}.itercoupling_{iterations_coupling}/"
    else:
        path = f"data.{model}.rewire_{local_rewiring}.N_{N}.l_{l}.tstep_{tstep}/" # where simulation data is saved
    
    if local_rewiring:
        path = path[:-1]+f'.thresh_{activity_threshold}/'
    
    if not os.path.exists(path):
        os.mkdir(path) # create path
    
    #if feedforward==True:
    #   np.savetxt(f'{path}adjacency.{model}.rewire_{local_rewiring}.N_{N}.l_{l}.tstep_{tstep}.csv', A, fmt='%i', delimiter=',')
    
        
    #print(f"\nN = {N}\ntstep = {tstep}\niterations = {iterations}\nmodel = {model}\nrewiring = {local_rewiring}\nactivity threshold = {activity_threshold}\nm = {m}\nS = {S}\n")
    global_vars = f'''

N = {N}
tstep = {tstep}
iterations = {iterations}
model = {model}
p_ne = {p_ne}

rewiring = {local_rewiring}
nr_neurons = {nr_neurons}
activity threshold = {activity_threshold}
batch_size = {batch_size}
add_factor = {add_factor}

load_external_couplings = {load_external_couplings}
iterations_coupling = {iterations_coupling}

torch_impl = {torch_impl}

m = {m}
S = {S}

'''
    print(global_vars)
    
    random_variables_rewire = np.empty((iterations,5*nr_neurons),float)
    
    sub_path_m = f"m_{m}/"
    sub_path_S = f"S_{S}/"
    
    if not os.path.exists(path+sub_path_m+sub_path_S):
        os.makedirs(path+sub_path_m+sub_path_S)

    shutil.copyfile(sys.argv[0], f'{path}{sub_path_m}{sub_path_S}{sys.argv[0]}') # save code
    shutil.copyfile(f'PARAM_{sys.argv[0]}', f'{path}{sub_path_m}{sub_path_S}LOG_{sys.argv[0]}') # save logfile
    
    # save for plotting
    sigma_dyn = np.zeros(iterations, float) 
    activity_ratio = np.zeros((iterations, len(n_l)-1), float)
    activities = np.zeros((iterations, len(n_l)), float)

    for iter in np.arange(iterations):
        
        # set up full path (number of iteration)
        if iter%500==0:
            num_path = f"{iter}-{iter+499}/"
            
            if not os.path.exists(path+sub_path_m+sub_path_S+num_path):
                os.mkdir(path+sub_path_m+sub_path_S+num_path)
            
            print(f"landmark iter = {iter}")
                
        #np.savetxt(f'{path}{sub_path_m}{sub_path_S}{num_path}W.{model}.rewire_{local_rewiring}.N_{N}.l_{l}.tstep_{tstep}.S_{S}.m_{m}.iter_{iter}.csv', W, delimiter=',')
        # initial conditions
        if feedforward == True:
            idx = np.arange(n_l[0])
            if local_rewiring == True:
                #S = 1#np.random.choice(range(1,n_l[0]+1),1)[0]
                batch_vec = np.zeros(batch_size, object)
                #trace_vec = np.zeros(batch_size, object)
            
        else:
            idx = np.arange(N)
            #if local_rewiring == True: # random initial conditions for local rewiring
            #   S = np.random.choice(range(1,N+1),1)[0]
        
        S_mem = False # memorize whether S is randomly chosen
        if S == 'random':
            S = np.random.choice(range(1,n_l[0]+1),1)[0]
            S_mem = True
        
        X = np.zeros((N,tstep), float)
        #initially_active_nodes = np.empty((iterations,batch_size,S),int)
        #initially_active_nodes = []        
        
        if local_rewiring == False:
            
            ic_idx = np.random.choice(idx,S,replace=False) # draw initially active neurons
            #initially_active_nodes[iter,:] = ic_idx # save initially active nodes
            mask = [True if i in ic_idx else False for i in range(N)] # create boolean array
            #X[mask,0] = np.random.random(sum(mask))*2 - 1 # set initial conditions, random numbers in [-1, 1)
            X[mask,0] = np.random.random(sum(mask)) # set in initial condition, random numbers in [0,1)
            
            if set_external_ic:
                X[:,0] = np.genfromtxt(path_to_ic, delimiter=',')[:,0]
            #np.savetxt(f'{path}/ic_states.N_{N}.S_{S}.m_{my_floor(m,2)}.tstep_{tstep}.iter_{iter}.csv', X[:,0], delimiter=',')
            
            if load_external_couplings == False:
                if model == "shew09":
                    # model: shew 09
                    
                    # mean and standard deviation of coupling weights
                    mean = 1/N
                    stdev = 1/N
                                
                    # construct random parameters
                    if feedforward==False:
                        statesave_RNG('W', iter, path+sub_path_m+sub_path_S+num_path) # save RNG seed
                        W = np.random.uniform((1-math.sqrt(3))/N,(1+math.sqrt(3))/N,(N,N)) # matrix of synaptic coupling strengths, with mean and SD ~1/N
                        zeta = np.random.uniform(0,1,(N,tstep)) # implements probabilistic nature of synapses (time- and node-dependent, != paper)
                        scaling_W = m
                        W = np.multiply(W, scaling_W)
                        #np.savetxt(f'{path}{sub_path_m}{sub_path_S}{num_path}W.N_{N}.l_{l}.tstep_{tstep}.S_{S}.m_{m}.iter_{iter}.csv', W, delimiter=',')
                        #np.savetxt(f'{path}{sub_path_m}{num_path}/zeta.N_{N}.S_{S}.m_{my_floor(m,2)}.step_{tstep}.FF_{feedforward}.iter_{iter}.csv', zeta, delimiter=',')
                    else:
                        # calculate couplings for each layer once, at beginning of each simulation
                        #all_W = np.array([construct_uniform_feedforward_couplings_layerwise(n_l, mean, stdev, layer) for layer in range(len(n_l)-1)])
                        #sum_W = np.sum(all_W)
                        
                        # calculate sparse coupling matrix
                        W = construct_uniform_feedforward_couplings_sparse(n_l, mean, stdev, path, sub_path_m, sub_path_S, num_path, iter)
                        scaling_W = m
                        W = W.multiply(scaling_W)       
                    
                else: # elif model == "meisel20":
                    # model meisel 20
                    
                    # mean and standard deviation of coupling weights
                    mean = 1/2
                    stdev = 1/np.sqrt(12)
                    
                    if feedforward==False:
                        W = np.random.uniform(0,1,(N,N)) # initial coupling weight distribution
                        scaling_W =  N*m/W.sum(axis=None)
                        W = np.multiply(W, scaling_W)
                        # NO INIBITORY LINKS!
                        #indices_inhibitory = np.random.choice(list(range(N)), math.floor(0.2*N), replace=False) # chose inhibitory neurons
                        #W[:,indices_inhibitory] = W[:,indices_inhibitory] * (-1) # set chosen neurons inhibitory
                    else:
                        # calculate couplings for each layer once, at beginning of each simulation
                        #all_W = np.array([construct_uniform_feedforward_couplings_layerwise(n_l, mean, stdev, layer) for layer in range(len(n_l)-1)])
                        #sum_W = np.sum(all_W)
                        
                        # calculate sparse coupling matrix
                        W = construct_uniform_feedforward_couplings_sparse(n_l, mean, stdev, path, sub_path_m, sub_path_S, num_path, iter)
                        scaling_W = N*m/W.sum(axis=None)
                        W = W.multiply(scaling_W)
                
            else: # load_external_couplings == True
                if iter == 0:
                    # load random variables from run with local_rewiring = True
                    random_variables_rewire = np.genfromtxt(f'/fast/users/vocks_c/scratch/binary-neuron-network/data.{model}.rewire_True.N_{N}.l_{l}.tstep_{tstep}/m_{m}/S_random/random_indices_rewire.meisel20.rewire_True.N_{N}.l_50.tstep_{tstep}.S_random.m_{m}.csv', delimiter=',')
                    for iter_coupling in np.arange(iterations_coupling):
                        if iter_coupling==0:
                            print(f'loading initial coupling matrix')
                            num_path = f"{iter}-{iter+499}/"
                            path_to_coupling_seed = f'/fast/users/vocks_c/scratch/binary-neuron-network/data.{model}.rewire_True.N_{N}.l_{l}.tstep_{tstep}/m_{m}/S_random/0-499/'
                            W = reconstruct_couplings_binary_neuron_network.reconstruct_couplings(iter_coupling, m, path_to_coupling_seed)
                        else:
                            print('reconstruct rewiring process')
                            #random_index, random_variable, coupling, average_act, inactive = random_variables_rewire[iter-1,:]
                            random_indices = random_variables_rewire[iter_coupling-1,:nr_neurons].astype(int)
                            random_variables = random_variables_rewire[iter_coupling-1,nr_neurons:2*nr_neurons].astype(int)
                            coupling = random_variables_rewire[iter_coupling-1,2*nr_neurons:3*nr_neurons]
                            average_act = random_variables_rewire[iter_coupling-1,3*nr_neurons:4*nr_neurons]
                            inactive = random_variables_rewire[iter_coupling-1,4*nr_neurons:5*nr_neurons]
                            W[random_indices, random_variables] = coupling
                            
                        if W.shape != (N,N):
                                raise TypeError('Dimension of coupling matrix (external) does not match parameters.')
            
            if torch_impl:
                #time1 = datetime.datetime.now()
                if iter == 0:
                    # if model == 'meisel20':
                    #     net = net_meisel20(n_l) # instantiating network
                    # elif model == 'relu':
                    #     net = net_relu(n_l)
                    # elif model == 'tanh':
                    #     net = net_tanh(n_l)
                    # elif model == 'shew09':
                    #     raise TypeError('Model Shew09 not implemented in torch network')
                    
                    if model == 'shew09':
                        raise TypeError('Model Shew09 not implemented in torch network')
                    else:
                        net = net_init(n_l, model)
                    
                net = custom_weights_to_tensor(W, net) # transfer coupling matrix (sparse matrix) to tensor in instantiated network
                net.to(device)
                ic = torch.Tensor(X[:n_l[0], 0]).to(torch.float64).to(device) # set up tensor for initial conditions
                
                #p_ne = torch.Tensor([p_ne])
                #print('computation iteration {}'.format(iter))

                output = net(ic[np.newaxis]) # forward pass
                #time2 = datetime.datetime.now()
                #print('sim time = {}'.format(time2-time1))
                #X = trace_to_statematrix(net.trace, n_l, tstep, ic) # save all layers
                #X = cut_statematrix(X) # cut dim = 1 if activity has ceased before tstep

                # changed behavior to save memory:
                # save trace, not full state vector
                #X = np.array([item.detach().cpu().numpy() for sublist in net.trace for item in sublist]).T

                layers_as_list = [item.detach().cpu().numpy() for sublist in net.trace for item in sublist]
                X = array_from_ragged_list(layers_as_list)
                #print('sample {} done'.format(sample))
            
            else: # no pytorch implementation:
                for t in range(1,tstep): # percolate each timestep
                   #print('t = {}'.format(t))
                   if np.sum(X[:,t-1]) != 0: # check if activity ceased in previous timestep
                       if model == "shew09":
                           X[:,t] = percolate_shew09(X,W,zeta,t-1,feedforward)
                       elif model == "meisel20":
                           X[:,t] = percolate_meisel20(X,W,p_ne,t-1,feedforward)
                   else: 
                       X = X[:,0:t] # only save activity
                   if X.shape[1] < tstep: # break t - loop if activity has ceased
                       break
            
        else: # local rewiring == True
            
            if iter == 0:
                
                if model == "shew09":
                    # model: shew 09
                    
                    mean = 1/N # mean of resulting coupling matrix
                    stdev = 1/N # standard deviation of original matrix without deleted links
                    
                    # construct random parameters
                    if feedforward==False:
                        statesave_RNG('W', iter, path+sub_path_m+sub_path_S+num_path) # save RNG seed
                        W = np.random.uniform((1-math.sqrt(3))/N,(1+math.sqrt(3))/N,(N,N)) # matrix of synaptic coupling strengths, with mean and SD ~1/N
                        zeta = np.random.uniform(0,1,(N,tstep)) # implements probabilistic nature of synapses (time- and node-dependent, != paper)
                        scaling_W = m # save for self_organize_couplings()
                        W = np.multiply(W, scaling_W)
                        
                        #np.savetxt(f'{path}{sub_path_m}{sub_path_S}{num_path}W.N_{N}.l_{l}.tstep_{tstep}.S_{S}.m_{m}.iter_{iter}.csv', W, delimiter=',')
                        #np.savetxt(f'{path}{sub_path_m}{num_path}/zeta.N_{N}.S_{S}.m_{my_floor(m,2)}.step_{tstep}.FF_{feedforward}.iter_{iter}.csv', zeta, delimiter=',')
                    else:
                        # calculate couplings for each layer once, at beginning of each simulation
                        #all_W = np.array([construct_uniform_feedforward_couplings_layerwise(n_l, mean, stdev, layer) for layer in range(len(n_l)-1)])
                        #sum_W = np.sum(all_W)
                        
                        # calculate sparse coupling matrix
                        W = construct_uniform_feedforward_couplings_sparse(n_l, mean, stdev, path, sub_path_m, sub_path_S, num_path, iter)
                        scaling_W = m
                        W = W.multiply(m)
                    
                elif model == "meisel20":
                    # model meisel 20
                    
                    mean = 1/2 # mean of resulting coupling matrix
                    stdev = 1/np.sqrt(12) # standard deviation of original couplings without deleted links (c=0)
                    
                    if feedforward==False:
                        statesave_RNG('W', iter, path+sub_path_m+sub_path_S+num_path) # save RNG seed
                        W = np.random.uniform(0,1,(N,N)) # initial coupling weight distribution
                        scaling_W = N*m/W.sum(axis=None)
                        W = np.multiply(W, scaling_W)
                        
                    else:
                        # calculate couplings for each layer once, at beginning of each simulation
                        #all_W = np.array([construct_uniform_feedforward_couplings_layerwise(n_l, mean, stdev, layer) for layer in range(len(n_l)-1)])
                        #sum_W = np.sum(all_W)
                        #time1 = datetime.datetime.now()
                        # calculate sparse coupling matrix
                        W = construct_uniform_feedforward_couplings_sparse(n_l, mean, stdev, path, sub_path_m, sub_path_S, num_path, iter)
                        scaling_W = N*m/W.sum(axis=None)
                        W = W.multiply(scaling_W)
                        #time2 = datetime.datetime.now()
                        #print('ic build time = {}'.format(time2-time1))
                else: # model ciresan weight init
                    # set weights when instantiating network
                    print()
            
            for sample in range(batch_size):
                if S_is_string:
                    S = np.random.choice(range(1,n_l[0]+1),1)[0]
                    ic_idx = np.random.choice(idx,S,replace=False) # draw initially active neurons
                else:
                    if batch_size > len(idx):
                        raise ValueError('Batch size is larger than number of available input nodes. Aborting.')
                    ic_idx = [idx[sample]] #np.random.choice(idx,S,replace=False) # draw initially active neurons
                
                #print('ic_idx = {}'.format(ic_idx))
                #initially_active_nodes[iter,sample,:] = ic_idx # save initially active nodes
                #initially_active_nodes += ic_idx
                mask = [True if i in ic_idx else False for i in range(N)] # create boolean array
                
                X[mask,0] = np.random.random(sum(mask))*2 - 1 # set initial conditions, random numbers in [-1, 1)
                #X[mask,0] = np.random.random(sum(mask))
                if torch_impl:
                    #time1 = datetime.datetime.now()
                    if iter == sample == 0:
                        # if model == 'meisel20':
                        #     net = net_meisel20(n_l)
                        # if model == 'relu':
                        #     net = net_relu(n_l)
                        # if model == 'tanh':
                        #     net = net_tanh(n_l)
                        # if model == 'shew09':
                        #     raise TypeError('Model Shew09 not implemented in torch network') 
                        
                        if model == 'shew09':
                            raise TypeError('Model Shew09 not implemented in torch network')
                        else:
                            # instantiate network
                            net = net_init(n_l, model)
                            
                            if model == 'tanh':
                                net = uniform_weights_to_tensor(net, low_bound=m*-0.05, high_bound=m*0.05) # set uniformly distributed weights
                            else:
                                # load custom weight matrix
                                net = custom_weights_to_tensor(W, net) # transfer coupling matrix (sparse matrix) to tensor in instantiated network
                            
                            # send to device
                            net.to(device)
                        
                    ic = torch.Tensor(X[:n_l[0], 0]).to(torch.float32).to(device) # set up tensor for initial conditions
                    
                    #p_ne = torch.Tensor([p_ne])
                    
                    output = net(ic[np.newaxis]) # forward pass
                    #print('forward pass done')
                    #time2 = datetime.datetime.now()
                    #print('sim time = {}'.format(time2-time1))
                    #batch_vec[sample] = trace_to_statematrix(net.trace, n_l, tstep, ic) # save all layers
                    #trace_vec[sample] = net.trace
                    #print('sample {} done'.format(sample))
                    #batch_vec[sample] = np.array([item.detach().cpu().numpy() for sublist in net.trace for item in sublist]).T
                    layers_as_list = [item.detach().cpu().numpy() for sublist in net.trace for item in sublist]
                    batch_vec[sample] = array_from_ragged_list(layers_as_list)
                    #print('first batch done')
                else: # no pytorch implementation
                    #time1 = datetime.datetime.now()
                    for t in range(1,tstep): # percolate each timestep
                        if np.sum(X[:,t-1]) != 0: # check if activity ceased in previous timestep
                            
                            if model == 'relu' or model == 'tanh':
                                raise RuntimeError(f'Model {model} not implemented in numpy. Switch to pytorch.')
                                
                            if model == "shew09":
                                X[:,t] = percolate_shew09(X,W,zeta,t-1,feedforward)
                            elif model == "meisel20":
                                X[:,t] = percolate_meisel20(X,W,p_ne,t-1,feedforward)
                        else: 
                            X = X[:,0:t] # only save activity
                        if X.shape[1] < tstep: # break t - loop if activity has ceased
                            break
                    #time2 = datetime.datetime.now()
                    #print('sim time = {}'.format(time2-time1))
                    if feedforward == False:
                        break

                    batch_vec[sample] = rebuild_statematrix(X, tstep)
                    
                    # renew state vector after each sample
                    X = np.zeros((N,tstep), float)
                    #S = np.random.choice(range(1,n_l[0]+1),1)[0]
                    #ic_idx = np.random.choice(idx,S,replace=False) # draw initially active neurons
                    #mask = [True if i in ic_idx else False for i in range(N)]
                    #X[mask,0] = True # set initial conditions
                
            else:
                X = np.sum(batch_vec) # sum results from whole batch
                for i in enumerate(np.sum(X, axis=0)):
                    if i[1] == 0:
                        X = X[:,:i[0]+1]
                
                # save \sigma_{dyn} for plotting
                #activities[iter, :], activity_ratio[iter, :], sigma_dyn[iter] = sigma_dynamic(rebuild_statematrix(X, tstep), skip_layer, n_l)
                
            # set integration times (bornholdt rohlf 2000)
            #T1 = 0
            #T2 = X.shape[1]
            #if T2 == tstep:
            #   T1 = round(T2/2)
            #time1 = datetime.datetime.now()
            
            # setting the weights 
            if torch_impl:
                if batch_size != 1:
                    RuntimeError('Only batch_size=1 is implemented in pytorch.')
                
                activations = net.trace
                weights = [layer.weight.detach() for _, layer in enumerate(net.linears)]
                #activations = [trace.cpu().numpy() for trace in net.trace]
                #weights = [layer.weight.detach().cpu().numpy() for layer in net.linears]
                
                # get indices chosen by rewiring rule
                post_indices, pre_indices, new_coupling_weights, average_activities, is_active = self_organize_couplings_torch(
                    activations, weights, 0, tstep, activity_threshold, nr_neurons, add_factor, n_l
                    )
                # copy indices tensors to host memory
                post_indices = post_indices.cpu().numpy()
                pre_indices = pre_indices.cpu().numpy()
                new_coupling_weights = new_coupling_weights.cpu().numpy()
                average_activities = average_activities.cpu().numpy()
                
                # update weights in network instance
                net = update_weights(net, post_indices, pre_indices, new_coupling_weights, np.array(n_l), device)
                
                # save for later inspection
                random_variables_rewire[iter,:] = np.concatenate([post_indices, pre_indices, new_coupling_weights, average_activities, is_active])
                
            else:
                W, random_variables = self_organize_couplings(rebuild_statematrix(X, tstep), W, 0, tstep, activity_threshold, nr_neurons, add_factor) # update W, save random variables
                #time2 = datetime.datetime.now()
                #print('rewiring time = {}'.format(time2-time1))
                random_variables_rewire[iter,:] = random_variables
                #print('iter = {}, mean(W) = {}'.format(iter, statistics.mean(np.concatenate(W.toarray()))))
            
        if S_mem:
            S = 'random' # rename S for filenames
        #np.savetxt(f'{path}/prod_term.N_{N}.S_{S}.m_{my_floor(m,2)}.tstep_{tstep}.csv', prod_term, delimiter=',')  
        np.savetxt(f'{path}{sub_path_m}{sub_path_S}{num_path}state_matrix.{model}.rewire_{local_rewiring}.N_{N}.l_{l}.tstep_{tstep}.S_{S}.m_{m}.iter_{iter}.csv', X, fmt='%f', delimiter=',') # save state matrix
    #initially_active_nodes = initially_active_nodes.reshape(-1,initially_active_nodes.shape[2])
    #np.savetxt(f'{path}{sub_path_m}{sub_path_S}initially_active_nodes.{model}.rewire_{local_rewiring}.N_{N}.l_{l}.tstep_{tstep}.S_{S}.m_{m}.csv', initially_active_nodes, fmt='%i', delimiter=',') # save initially active nodes
    
    if local_rewiring == True:
        np.savetxt(f'{path}{sub_path_m}{sub_path_S}random_indices_rewire.{model}.rewire_{local_rewiring}.N_{N}.l_{l}.tstep_{tstep}.S_{S}.m_{m}.csv', random_variables_rewire, delimiter=',') # save random indices from rewire process
        #np.savetxt(f'{path}{sub_path_m}{sub_path_S}sigma_dyn.{model}.rewire_{local_rewiring}.N_{N}.l_{l}.tstep_{tstep}.S_{S}.m_{m}.csv', sigma_dyn, delimiter=',')
        #np.savetxt(f'{path}{sub_path_m}{sub_path_S}activity_ratio.{model}.rewire_{local_rewiring}.N_{N}.l_{l}.tstep_{tstep}.S_{S}.m_{m}.csv', activity_ratio, delimiter=',')
        #np.savetxt(f'{path}{sub_path_m}{sub_path_S}activity.{model}.rewire_{local_rewiring}.N_{N}.l_{l}.tstep_{tstep}.S_{S}.m_{m}.csv', activities, delimiter=',')
    print('Simulation ended. Results saved to {}'.format(path))
    
iterate_percolation_process(S,tstep,iterations,n_l, activity_threshold)

end_time = datetime.datetime.now()
elapsed_time = end_time - start_time
print('Time elapsed (hh:mm:ss.ms) {}'.format(elapsed_time))
