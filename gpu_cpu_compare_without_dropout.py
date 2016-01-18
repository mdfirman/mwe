# System and core utils
import numpy as np
import sys

# Theano and lasagne
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer

###################################
# Build the network
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
domain_target_var = T.ivector('domain_targets')
num_classes = 2

net = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

halfway_net = DenseLayer(net, num_units=1024)

net = DenseLayer(halfway_net,
                 num_units=num_classes,
                 nonlinearity=lasagne.nonlinearities.softmax)

# gradient reversal branch
gr_branch = DenseLayer(halfway_net, num_units=512)
gr_branch = DenseLayer(gr_branch,
                       num_units=2,
                       nonlinearity=lasagne.nonlinearities.softmax)

###################################
# Define and compile Theano

# I.e., you've got one output layer for the source task classification, and another output layer for the domain classification, and both share the same input layer (and a part of the network).
# You'd then define two ordinary loss functions:
pred_sourcetask, pred_domainclass = lasagne.layers.get_output(
    [net, gr_branch])
loss_sourcetask = lasagne.objectives.categorical_crossentropy(
    pred_sourcetask, target_var).mean()
loss_domainclass = lasagne.objectives.categorical_crossentropy(
    pred_domainclass, domain_target_var).mean()

# And use that to update the networks as done in the paper:
params1 = lasagne.layers.get_all_params(net)
params2 = lasagne.layers.get_all_params(gr_branch)
common = set(params1) & set(params2)

# hp_lambda = 0.1
updates1 = lasagne.updates.nesterov_momentum(
    loss_sourcetask - 0.1 * loss_domainclass, params1,
    learning_rate=0.0001, momentum=0.9)

updates2 = lasagne.updates.nesterov_momentum(
    loss_domainclass, list(set(params2) - common),
    learning_rate=0.0001, momentum=0.9)

# updates1.update(updates2)

def inspect_inputs(i, node, fn):
    for idx, inputt in enumerate(fn.inputs):
        if np.isnan(inputt[0]).sum():
            print "Found a nan in inputs - quitting"
            print inputt, idx
            quit()

class_train = theano.function(
    [input_var, target_var, domain_target_var],
    [], updates=updates1,
    mode=theano.compile.MonitorMode(
        pre_func=inspect_inputs).excluding('local_elemwise_fusion', 'inplace')
)

#######################################
# Performing training

np.random.seed(10)
batch_size = 50
num_batches = 100

for batch_idx in range(num_batches):

    print batch_idx,
    sys.stdout.flush()

    # random batch each time
    xx = np.random.rand(batch_size, 3, 32, 32).astype(np.float32)
    yy = np.random.randint(0, 1, batch_size).astype(np.int32)
    dmn_yy = np.random.randint(0, 1, batch_size).astype(np.int32)

    class_train(xx, yy, dmn_yy)

print "Exited succesfully"
