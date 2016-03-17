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

import numpy as np
import scipy as sp
from utils import *


class stochastic_layer():

    def __init__(self, M,D, binaryF, numpy_rng):

        self.init_weight(M, D, binaryF, numpy_rng)
        pass

    def init_weight(self, M, D, binaryF, numpy_rng):
        
        if binaryF:
            self.R      = initialize_weight(M, D, 'R', numpy_rng, 'uniform')
            self.rbias  = theano.shared(np.zeros((D), dtype=theano.config.floatX), name='rbias')
            self.params = [self.R, self.rbias]
            
        else:
            self.R_mu    = initialize_weight(M, D, 'R_mu', numpy_rng, 'uniform')
            self.rmubias = theano.shared(np.zeros((D), dtype=theano.config.floatX), name='rmubias')

            self.R_sig   = initialize_weight(M, D, 'R_sig', numpy_rng, 'uniform')
            self.rsigbias= theano.shared(np.zeros((D), dtype=theano.config.floatX), name='rsigbias')
            self.params = [self.R_mu, self.rmubias, self.R_sig, self.rsigbias]


    def propagate(self,H, binaryF):

        if binaryF:
            return  T.nnet.sigmoid(T.dot(H, self.R) + self.rbias)
        else:
            return T.nnet.sigmoid(T.dot(H, self.R_mu) + self.rmubias), \
                    T.nnet.softplus(T.dot(H, self.R_sig) + self.rsigbias) + 0.0001






