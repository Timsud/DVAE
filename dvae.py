''' Version 1.000
 Code provided by Daniel Jiwoong Im
 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''

'''Demo of Denoising Criterion for Variational Auto-encoding Framework.
For more information, see :http://arxiv.org/abs/1511.06406
'''
import os
import sys

import theano
import theano.tensor as T
from stochastic_layer import *
from utils import *
import numpy as np
import scipy as sp
import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState()
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))


class DVAE2(object):

    def __init__(self, model_params):

        [self.batch_sz, self.num_dim, self.num_hids, numpy_rng, self.dim_sample, binaryF]  = model_params 

        self.numpy_rng = numpy_rng
        self.init_params(numpy_rng) 
        self.last_layer = stochastic_layer(self.num_hids[0], self.num_dim, binaryF, numpy_rng)  
        self.params = self.params + self.last_layer.params

        self.MRG = RNG_MRG.MRG_RandomStreams(numpy_rng.randint(2 ** 30))


    def init_params(self, numpy_rng):
        """Initialize Weight parameters"""
            
        self.W1     = initialize_weight(self.num_dim, self.num_hids[0], 'W1', numpy_rng, 'uniform')
        self.W2     = initialize_weight(self.num_hids[0], self.num_hids[0], 'W2', numpy_rng, 'uniform')
        self.hbias1 = theano.shared(np.zeros((self.num_hids[0]), dtype=theano.config.floatX), name='hbias1')
        self.hbias2 = theano.shared(np.zeros((self.num_hids[0]), dtype=theano.config.floatX), name='hbias2')

        self.W_Qmean = initialize_weight(self.num_hids[0], self.dim_sample,  'W_Qmean', numpy_rng, 'uniform') 
        self.W_Qvar  = initialize_weight(self.num_hids[0], self.dim_sample,  'W_Qvar', numpy_rng, 'uniform')
        self.Qm_bias = theano.shared(np.zeros((self.dim_sample,), dtype=theano.config.floatX), name='Qm_bias')
        self.Qv_bias = theano.shared(np.zeros((self.dim_sample,), dtype=theano.config.floatX), name='Qv_bias')

        self.W_h_z  = initialize_weight(self.dim_sample, self.num_hids[0], 'W_h_z', numpy_rng, 'uniform')
        self.h_z_bias  = theano.shared(np.zeros((self.num_hids[0],), dtype=theano.config.floatX), name='h_z_bias')
        self.params = [self.W1, self.W2, self.hbias1, self.hbias2, self.W_h_z ,self.h_z_bias, \
                                        self.W_Qmean, self.W_Qvar, self.Qm_bias, self.Qv_bias]


    def get_hidden_enc(self, X):
        """ Returns the output of neural network in inference network (just before the stochastic output layer) """
        
        H1 = T.nnet.softplus(T.dot(X, self.W1) + self.hbias1) 
        return T.nnet.softplus(T.dot(H1, self.W2) + self.hbias2) 


    def get_hidden_dec(self, Z):
        """ Returns the output of neural network in generator network (just before the stochastic output layer) """
        return T.nnet.softplus(T.dot(Z, self.W_h_z) + self.h_z_bias)


    def sampleZ_cond_H(self, H, K):
        """ Return samples from latent variable representation layer """
            
        sigs = MRG.normal(size=(K, self.dim_sample), avg=0., std=1.)
        mean =  T.dot(H, self.W_Qmean) + self.Qm_bias
        var  = (T.dot(H, self.W_Qvar)  + self.Qv_bias)
        return mean + T.exp(var) * sigs


    def KL_Q_P(self,H):
        """ Returns KL divergence KL(q(z|x)||p(z)) """

        mean    = T.dot(H, self.W_Qmean) +self.Qm_bias
        log_sig = (T.dot(H, self.W_Qvar) +self.Qv_bias)
        kl =  T.sum(- log_sig + 0.5 * (T.exp(2*log_sig) + mean**2) - 0.5, axis=-1)
        return kl


    def get_recon_H(self, H, binaryF=True):
        """ returns the reconstructed input given the decoder hidden units """

        return self.last_layer.propagate(H,binaryF)


    def get_recon_X(self, X, binaryF=True):
        """ returns the reconstructed input."""

        H_enc   = self.get_hidden_enc(X)
        Z       = self.sampleZ_cond_H(H_enc, self.batch_sz)
        H_dec   = self.get_hidden_dec(Z)
        if binaryF:
            return self.get_recon_H(H_dec, binaryF=binaryF)
        else:
            RX_mu, RX_sig   = self.get_recon_H(H_dec, binaryF=binaryF)
            return RX_mu

        return RX, H_enc


    def cost(self, X, corrupt_in=0, binaryF=True, CrossEntropyF=True, ntype='salt_pepper'):
        """ This function computes the cost and the updates for one trainng
        step """

        CX      = get_corrupted_input(self.numpy_rng, X, corrupt_in, ntype=ntype)
        H_enc   = self.get_hidden_enc(CX)
        Z       = self.sampleZ_cond_H(H_enc, self.batch_sz)
        H_dec   = self.get_hidden_dec(Z)


        if binaryF:
            RX    = self.get_recon_H(H_dec, binaryF=binaryF)
            lossX = T.sum(T.nnet.binary_crossentropy(RX, X), axis=1)
        else:
            RX_mu, RX_sig   = self.get_recon_H(H_dec, binaryF=binaryF)
            lossX           = 0.5 * T.sum(T.sqr(X - RX_mu) / RX_sig**2 + 2 * T.log(RX_sig) + T.log(2 * np.pi), axis=1)


        logZ    = self.KL_Q_P(H_enc)
        J       = (T.mean(lossX) + T.mean(logZ))
        return J


    def get_sample(self, num_sam, binaryF):
        """ Return samples drawn from p(z) and propagating through p(x|z) """

        Z = self.MRG.normal(size=(num_sam, self.dim_sample), avg=0., std=1.)
        H_dec   = self.get_hidden_dec(Z)
        if binaryF:
            return self.get_recon_H(H_dec, binaryF=binaryF)
        else:
            RX_mu, RX_sig   = self.get_recon_H(H_dec, binaryF=binaryF)
            return RX_mu


    def weight_decay(self):
        """ L2 weight decay """

        return (self.W ** 2).sum() /2



