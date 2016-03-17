import os,sys
import numpy as np
import scipy as sp
import theano
import theano.tensor as T
import pickle,cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip

from utils import *
import timeit, time
from optimize import *
from dvae import *

import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState()
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))

datapath='/groups/branson/home/imd/Documents/machine_learning_uofg/data/MNIST/'

if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/figs/"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/figs/")
if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/params/"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/params/")

def lets_train(train_set, valid_set, test_set, opt_params, model_params, model_type, opt_method='ADAM'):


    [batch_sz, epsilon, momentum, num_epoch, N, Nv, Nt, binaryF, crossEntropyF, corrupt_in, ntype]   = opt_params
    [batch_sz, D, num_hids, rng, num_z, binaryF]  = model_params

    opt_params = [batch_sz, epsilon, momentum, num_epoch, N, Nv, Nt, corrupt_in, ntype]

    # compute number of minibatches for training, validation and testing
    num_batch_train = N  / batch_sz
    num_batch_valid = Nv / batch_sz
    num_batch_test  = Nt / batch_sz

    vae = DVAE2(model_params)
    opt = Optimize(opt_params)

    if opt_method=='MGD':
        train_model, update_momentum, get_valid_cost \
                                        = opt.MGD(vae, train_set, valid_set, test_set, binaryF, crossEntropyF)
    else:
        train_model, get_valid_cost, get_test_cost = opt.ADAM(vae, train_set, valid_set, test_set, binaryF, crossEntropyF)


    get_samples = opt.get_samples(vae, binaryF)

    best_vl = np.infty
    k=0
    constant=3
    bookkeeping_vl = []
    for epoch in xrange(num_epoch+1):

        costs=[]
        if constant**k == epoch+1: 
            eps = epsilon * get_epsilon_decay(k+1, num_epoch, constant)
            k+=1

        exec_start = timeit.default_timer()
        for batch_i in xrange(num_batch_train):

            if opt_method=='MGD': update_momentum()
            cost_i = train_model(batch_i, lr=eps)
            if not np.isnan(cost_i): 
                costs.append(cost_i)

        exec_finish = timeit.default_timer()
        if  epoch==0:
            print 'Exec Time %f ' % ( exec_finish - exec_start)

        if epoch % 50 == 0 or epoch < 2 or epoch == (num_epoch-1):
            costs_vl = []
            for batch_j in xrange(num_batch_valid):
                cost_vl_j = get_valid_cost(batch_j)
                costs_vl.append(cost_vl_j)

            cost_vl = np.mean(np.asarray(costs_vl))
            cost_tr = np.mean(np.asarray(costs))
            bookkeeping_vl.append(cost_vl)
            print 'Epoch %d, epsilon %g, train cost %g, valid cost %g' % (epoch, eps, cost_tr, cost_vl)

            if best_vl > cost_vl and epoch > 0.4*num_epoch:
                best_vl = cost_vl
                save_the_weight(vae.params, './params/'+ model_type +'_param_mnist')

                costs_te = []
                for batch_j in xrange(num_batch_test):
                    cost_te_j = get_test_cost(batch_j)
                    costs_te.append(cost_te_j)

                cost_te = np.mean(np.asarray(costs_te))
                print '*** Epoch %d, test cost %g ***' % (epoch, cost_te)

    costs_te = []
    for batch_j in xrange(num_batch_test):
        cost_te_j = get_test_cost(batch_j)
        costs_te.append(cost_te_j)

    cost_te = np.mean(np.asarray(costs_te))
    print '*** Epoch %d, test cost %g ***' % (epoch, cost_te)

    return vae, cost_te



##################
## Hyper-params ##
##################
batch_sz      = 100
epsilon       = 0.0001
momentum      = 0.0
num_epoch     = 5000
num_hid       = 200
num_z         = 50
num_class     = 10
num_trial     = 1
binaryF       = True
CrossEntropyF = True
corrupt_in    = 0.0
model_type    = 'dvae'
rnn_type      = 'grnn'
data_type     = 'hugo_mnist'
print 'model type: ' + model_type
print 'rnn type: ' + rnn_type
print 'data Type: ' + data_type
ntype         = 'salt_pepper'

if __name__ == '__main__':

    #Note that there are two different binary MNIST in the literature. 
    if data_type == 'hugo_mnist':
        dataset = datapath+'/binarized_mnist.npz'
        data    = np.load(dataset)
        train_set = [data['train_data'], np.zeros(int(data['train_length']),)] 
        valid_set = [data['valid_data'], np.zeros(int(data['valid_length']),)] 
        test_set  = [data['test_data'] , np.zeros(int(data['test_length' ]),)]  
        N ,D = train_set[0].shape; Nv,D = valid_set[0].shape; Nt,D = test_set[0].shape
        
        train_set = shared_dataset(train_set)
        valid_set = shared_dataset(valid_set)
        test_set  = shared_dataset(test_set )

    else:
        dataset = datapath+'/mnist.pkl.gz'
        f = gzip.open(dataset, 'rb')
        train_set_o, valid_set_o, test_set_o = cPickle.load(f)
        N ,D = train_set_o[0].shape
        Nv,D = valid_set_o[0].shape
        Nt,D = test_set_o[0].shape
        f.close()
    print 'N, D %d %d' % (N,D)

    book_keeping = []
    for i in xrange(num_trial):
        if data_type != 'hugo_mnist':
            train_set = [rng.binomial(1, train_set_o[0]), train_set_o[1]] 
            valid_set = [rng.binomial(1, valid_set_o[0]), valid_set_o[1]] 
            test_set  = [rng.binomial(1,  test_set_o[0]),  test_set_o[1]] 

            train_set = ld.shared_dataset(train_set)
            valid_set = ld.shared_dataset(valid_set)
            test_set  = ld.shared_dataset(test_set )

        print 'batch sz %d, epsilon %g, num_hid %d, num_z %d,  num_epoch %d corrupt_in %g' % \
                                (batch_sz, epsilon, num_hid, num_z, num_epoch, corrupt_in)

        num_hids = [num_hid]
        opt_params   = [batch_sz, epsilon, momentum, num_epoch, N, Nv, Nt, binaryF, CrossEntropyF, corrupt_in, ntype]
        model_params = [batch_sz, D, num_hids, rng, num_z, binaryF]
        vae, cost_te = lets_train(train_set, valid_set, test_set, opt_params, model_params, model_type )
        book_keeping.append(cost_te)

    book_keeping = np.asarray(book_keeping)
    print '+++ Mean test nll %g std test nll %g +++' % (np.mean(book_keeping), np.std(book_keeping))

    f.close()

