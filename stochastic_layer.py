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






