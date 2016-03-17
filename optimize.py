import os, sys
import numpy as np
import pylab
import cPickle

import theano
import theano.tensor as T
import theano.tensor.signal.conv
from theano.tensor.shared_randomstreams import RandomStreams

import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState(1234)
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))


class Optimize():

    def __init__(self, opt_params):

        self.batch_sz, self.epsilon, self.momentum, self.num_epoch, \
                        self.N, self.Nv, self.Nt, self.corrupt_in, self.ntype = opt_params


    def MGD(self, model, train_set, valid_set, test_set, binaryF, CrossEntropyF):

        update_grads = []; updates_mom = []; deltaWs = {}
        mom = T.scalar('mom'); i = T.iscalar('i'); lr = T.fscalar('lr');
        X = T.matrix('X')
        X.tag.test_value = np.zeros((100, 560), dtype=theano.config.floatX)

        cost         = model.cost(X, binaryF=binaryF, CrossEntropyF=CrossEntropyF, corrupt_in=self.corrupt_in, ntype=self.ntype)
        cost_test    = model.cost(X, binaryF=binaryF, CrossEntropyF=CrossEntropyF, M=1, L=1)
        gparams      = T.grad(cost, model.params)
        #gparams      = self.clip_gradient(model.params, gparams)

        #Update momentum
        for param in model.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            deltaWs[param] = theano.shared(init)

        for param in model.params:
            updates_mom.append((param, param + deltaWs[param] * \
                            T.cast(mom, dtype=theano.config.floatX)))

        for param, gparam in zip(model.params, gparams):

            deltaV = T.cast(mom, dtype=theano.config.floatX)\
                    * deltaWs[param] - gparam * T.cast(lr, dtype=theano.config.floatX)     #new momentum

            update_grads.append((deltaWs[param], deltaV))
            new_param = param + deltaV

            update_grads.append((param, new_param))


        update_momentum = theano.function([theano.Param(mom,default=self.momentum)],\
                                                        [], updates=updates_mom)

        train_update    = theano.function([i, theano.Param(lr,default=self.epsilon),\
                theano.Param(mom,default=self.momentum)], outputs=cost, updates=update_grads,\
                givens={ X:train_set[0][i*self.batch_sz:(i+1)*self.batch_sz]})

        get_valid_cost   = theano.function([i], outputs=cost_test,\
                givens={ X:valid_set[0][i*self.batch_sz:(i+1)*self.batch_sz]})

        get_test_cost    = theano.function([i], outputs=cost_test,\
                givens={ X:test_set[0][i*self.batch_sz:(i+1)*self.batch_sz]})

        return train_update, update_momentum, get_valid_cost, get_test_cost


    def ADAM(self, model, train_set, valid_set, test_set, binaryF, CrossEntropyF,\
            beta1 = 0.1,beta2 = 0.001,epsilon = 1e-8, l = 1e-8):

        i = T.iscalar('i'); lr = T.fscalar('lr');
        X = T.matrix('X');
        X.tag.test_value = np.zeros((100, 560), dtype=theano.config.floatX)

        cost        = model.cost(X, binaryF=binaryF, CrossEntropyF=CrossEntropyF, corrupt_in=self.corrupt_in, ntype=self.ntype)
        cost_test   = model.cost(X, binaryF=binaryF, CrossEntropyF=CrossEntropyF)
        gparams     = T.grad(cost, model.params)
        gparams     = self.clip_gradient(model.params, gparams)

        '''ADAM Code from
            https://github.com/danfischetti/deep-recurrent-attentive-writer/blob/master/DRAW/adam.py
        '''
        self.m = [theano.shared(name = 'm', \
                value = np.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in model.params]
        self.v = [theano.shared(name = 'v', \
        	value = np.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in model.params]
        self.t = theano.shared(name = 't',value = np.asarray(1).astype(theano.config.floatX))
        updates = [(self.t,self.t+1)]

        for param, gparam,m,v in zip(model.params, gparams, self.m, self.v):

            b1_t = 1-(1-beta1)*(l**(self.t-1))
            m_t = b1_t*gparam + (1-b1_t)*m
            updates.append((m,m_t))
            v_t = beta2*(gparam**2)+(1-beta2)*v
            updates.append((v,v_t))
            m_t_bias = m_t/(1-(1-beta1)**self.t)
            v_t_bias = v_t/(1-(1-beta2)**self.t)
            new_param = param - lr*m_t_bias/(T.sqrt(v_t_bias)+epsilon)
            updates.append((param, new_param))

        #import ipdb; ipdb.set_trace()
        train_update = theano.function([i, theano.Param(lr,default=self.epsilon)],\
                outputs=cost, updates=updates,\
                givens={ X:train_set[0][i*self.batch_sz:(i+1)*self.batch_sz]})

        get_valid_cost   = theano.function([i], outputs=cost_test,\
                givens={ X:valid_set[0][i*self.batch_sz:(i+1)*self.batch_sz]})

        get_test_cost    = theano.function([i], outputs=cost_test,\
                givens={ X:test_set[0][i*self.batch_sz:(i+1)*self.batch_sz]})

        return train_update, get_valid_cost, get_test_cost


    def clip_gradient(self, params, gparams, scalar=5, check_nanF=True):
        """
            Sequence to sequence
        """
        num_params = len(gparams)
        g_norm = 0.
        for i in xrange(num_params):
            gparam = gparams[i]
            g_norm += (gparam**2).sum()
        if check_nanF:
            not_finite = T.or_(T.isnan(g_norm), T.isinf(g_norm))
        g_norm = T.sqrt(g_norm)
        scalar = scalar / T.maximum(scalar, g_norm)
        if check_nanF:
            for i in xrange(num_params):
                param = params[i]
                gparams[i] = T.switch(not_finite, 0.1 * param, gparams[i] * scalar)
        else:
            for i in xrange(num_params):
                gparams[i]  = gparams[i] * scalar

        return gparams


    def get_recon(self, model, valid_set, binaryF):

        i = T.iscalar('i');
        X = T.fmatrix('X')

        reconX = model.get_recon_X(X, binaryF=binaryF)
        get_recon   = theano.function([i], reconX, givens={ X:valid_set[0][:i]})
        return get_recon


    def get_samples(self, model, binaryF):

        i = T.iscalar('i');
        return theano.function([i], model.get_sample(i, binaryF))


