import os,sys
import numpy as np
import scipy as sp
import theano
import theano.tensor as T
import pickle,cPickle
import matplotlib
import gzip
import timeit, time
import scipy.io

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optimize import *
from dvae import *
from utils import *

import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState(1234)
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))

datapath='/groups/branson/home/imd/Documents/machine_learning_uofg/data/FrayFace'

if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/figs/ff"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/figs/ff")
if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/params/"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/params/")



def lets_train(train_set, valid_set, test_set, opt_params, model_params, model_type, ith_trial=None, opt_method='ADAM'):

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

    #get_reconX  = opt.get_recon(vae, valid_set, binaryF)
    get_samples = opt.get_samples(vae, binaryF)

    best_vl = np.infty
    for epoch in xrange(num_epoch+1):

        costs=[]
        eps = get_epsilon(epsilon, num_epoch, epoch)
        exec_start = timeit.default_timer()
        for batch_i in xrange(num_batch_train):

            if opt_method=='MGD': update_momentum()
            cost_i = train_model(batch_i, lr=eps)
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
            print 'Epoch %d, lr %g, tr cost %g, vl cost %g' % (epoch, eps, cost_tr, cost_vl)

            if best_vl > cost_vl and epoch > 0.4*num_epoch:
                best_vl = cost_vl
                save_the_weight(vae, './params/'+model_type+'_ff_z2_200')

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
epsilon       = 0.002
momentum      = 0.0
num_epoch     = 3000
num_z         = 10
num_class     = 10
num_trial     = 1
corrupt_in    = 0.025
num_hids      = [100,100]
binaryF       = False
CrossEntropyF = False
model_type    = 'dvae2'
ntype         = 'gaussian'

if __name__ == '__main__':

    dataset = datapath+'/frey_rawface.mat'
    data = scipy.io.loadmat(dataset)['ff'].T / 255.0
    perms = np.random.permutation(data.shape[0]) 
    data = data[perms]

    N_tr = 1572
    N_vl = 295   
    N_te = 200 

    train_set_o = [data[:N_tr], np.zeros((N_tr,))]
    valid_set_o = [data[:N_vl], np.zeros((N_vl,))]
    test_set_o  = [data[:N_te], np.zeros((N_te,))]

    N ,D = train_set_o[0].shape
    Nv,D = valid_set_o[0].shape
    Nt,D = test_set_o[0].shape

    best_hyp  = np.infty
    best_mean = np.infty
    best_std  = np.infty

    book_keeping = []
    for i in xrange(num_trial):

        train_set = shared_dataset(train_set_o)
        valid_set = shared_dataset(valid_set_o)
        test_set  = shared_dataset(test_set_o )

        print 'batch sz %d, epsilon %g, num_hid %d, num_z %d,  num_epoch %d corrupt_in %g' % \
                                (batch_sz, epsilon, num_hids[0], num_z, num_epoch, corrupt_in)

        opt_params   = [batch_sz, epsilon, momentum, num_epoch, N_tr, N_vl, N_te, binaryF, CrossEntropyF, corrupt_in, ntype]
        model_params = [batch_sz, D, num_hids, rng, num_z, binaryF]
        vae, cost_te = lets_train(train_set, valid_set, test_set, opt_params, model_params, model_type, ith_trial=i)
        book_keeping.append(cost_te)

    book_keeping = np.asarray(book_keeping)
    mean = np.mean(book_keeping)
    std  = np.std(book_keeping)
    print '+++ Mean test nll %g std test nll %g +++' % (mean, std )

    if best_mean > mean:
        best_mean = mean
        best_std  = std
        best_hyp = num_epoch

    print "****** best mean, std, hyper search ******" 
    print best_mean, best_std, best_hyp

