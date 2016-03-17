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
import time, timeit
import os,sys
import numpy as np
import scipy as sp
import PIL
import theano
import theano.tensor as T
import pickle,cPickle
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks
import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState()
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))
c = - T.constant(np.log(2*np.pi)).astype(theano.config.floatX)
c.tag.test_value = np.log(2*np.pi).astype(theano.config.floatX)

def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data 



def repmat_vec(x,k):

    return T.tile(x.dimshuffle([0,'x']), [1,k]).T

def repmat_mat(x,k):

    return T.tile(x.dimshuffle([0,1, 'x']), [1,1,k]).T

def repmat_tensor(x,k):

    return T.tile(x.dimshuffle([0,1, 2,'x']), [1,1,1,k]).T


def get_epsilon_decay(i, num_epoch, constant=4): 
    c = np.log(num_epoch/2)/ np.log(constant)
    return 10.**(1-(i-1)/(float(c)))

'''decaying learning rate'''
def get_epsilon(epsilon, n, i):
    return epsilon / ( 1 + i/float(n))


def log_likelihood_samplesImean_sigma(samples, mean, logvar):
    return c*T.cast(samples.shape[1], 'float32') /2  - \
               T.sum(T.sqr((samples-mean)/T.exp(logvar)) + 2*logvar, axis=1) / 2

def prior_z(samples):
    return c*T.cast(samples.shape[1], 'float32')/2 - T.sum(T.sqr(samples), axis=1) / 2


def log_likelihood_samplesImean_sigma2(samples, mean, logvar):
    return c*T.cast(samples.shape[2], 'float32') /2  - \
               T.sum(T.sqr((samples-mean)/T.exp(logvar)) + 2*logvar, axis=2) / 2

def prior_z2(samples):
    return c*T.cast(samples.shape[2], 'float32')/2 - T.sum(T.sqr(samples), axis=2) / 2


def log_mean_exp(x, axis):
    m = T.max(x, axis=axis, keepdims=True)
    return m + T.log(T.mean(T.exp(x - m), axis=axis, keepdims=True))



def normalize(data, vdata=None, tdata=None):
    mu   = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    data = ( data - mu ) / std

    if vdata == None and tdata != None:
        tdata = (tdata - mu ) /std
        return data, tdata, mu

    if vdata != None and tdata != None:
        vdata = (vdata - mu ) /std
        tdata = (tdata - mu ) /std
        return data, vdata, tdata, mu
    return data, mu



def get_corrupted_input(rng, input, corruption_level, ntype='zeromask', input_shape=None):

    if input_shape is None:
        input_shape = input.shape

    #theano_rng = RandomStreams()
    if corruption_level == 0.0:
        return input

    if ntype=='zeromask':
        return MRG.binomial(size=input_shape, n=1, p=1-corruption_level,dtype=theano.config.floatX) * input
    elif ntype=='gaussian':
        return input + MRG.normal(size = input_shape, avg = 0.0,
                std = corruption_level, dtype = theano.config.floatX)
    elif ntype=='salt_pepper':

        # salt and pepper noise
        print 'DAE uses salt and pepper noise'
        a = MRG.binomial(size=input_shape, n=1,\
                p=1-corruption_level,dtype=theano.config.floatX)
        b = MRG.binomial(size=input_shape, n=1,\
                p=corruption_level,dtype=theano.config.floatX)

        c = T.eq(a,0) * b
        return input * a + c

def salt_peper_noise(rng, input, corruption_dist):

    # salt and pepper noise
    print 'DAE uses salt and pepper noise'
    a = MRG.binomial(size=input.shape, n=1,\
            p=1-corruption_dist,dtype=theano.config.floatX)
    b = MRG.binomial(size=input.shape, n=1,\
            p=corruption_dist,dtype=theano.config.floatX)

    c = T.eq(a,0) * b
    return input * a + c

def get_corrupted_input3D(rng, input, corruption_level, ntype='zeromask', input_shape=None):

    if input_shape is None:
        input_shape = (input.shape[0], 1, input.shape[1])

    #theano_rng = RandomStreams()
    if corruption_level == 0.0:
        return input.dimshuffle([0,'x',1])

    if ntype=='zeromask':
        return MRG.binomial(size=input_shape, n=1, p=1-corruption_level,dtype=theano.config.floatX) * input.dimshuffle([0,'x',1])
    elif ntype=='gaussian':
        return input.dimshuffle([0,'x',1]) + MRG.normal(size = input_shape, avg = 0.0,
                std = corruption_level, dtype = theano.config.floatX)
    elif ntype=='salt_pepper':

        # salt and pepper noise
        print 'DAE uses salt and pepper noise'
        a = MRG.binomial(size=input_shape, n=1,\
                p=1-corruption_level,dtype=theano.config.floatX)
        b = MRG.binomial(size=input_shape, n=1,\
                p=corruption_level,dtype=theano.config.floatX)

        c = T.eq(a,0) * b
        return input.dimshuffle([0,'x',1]) * a + c


def special_SaP_noise_4_jyc(rng, input, corruption_level):

    # salt and pepper noise
    print 'DAE uses salt and pepper noise'
    a = MRG.binomial(size=input.shape, n=1,\
            p=1-corruption_level,dtype=theano.config.floatX)
    b = MRG.binomial(size=input.shape, n=1,\
            p=corruption_level,dtype=theano.config.floatX)

    c = T.eq(a,0) * b
    mask = - a + c
    CX = input * a + c
    return T.stacklists([X, noise_mask])

def save_the_weight(x,fname):
    f = file(fname+'.save', 'wb')
    cPickle.dump(x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    #When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue

    return shared_x, T.cast(shared_y, 'int32')


def initialize_weight(n_vis, n_hid, W_name, numpy_rng, rng_dist):

    if 'uniform' in rng_dist:
        W = numpy_rng.uniform(low=-np.sqrt(6. / (n_vis + n_hid)),\
                high=np.sqrt(6. / (n_vis + n_hid)),
                size=(n_vis, n_hid)).astype(theano.config.floatX)
        #if 'exp' in rng_dist :
        #    W = np.exp(-W)
    elif rng_dist == 'normal':
        W = 0.01 * numpy_rng.normal(size=(n_vis, n_hid)).astype(theano.config.floatX)
    elif rng_dist == 'ortho':
        N_ = int(n_vis / float(n_hid))
        sz = np.minimum(n_vis, n_hid)
        W = np.zeros((n_vis, n_hid), dtype=theano.config.floatX)
        for i in xrange(N_):
            temp = 0.01 * numpy_rng.normal(size=(sz, sz)).astype(theano.config.floatX)
            W[:, i*sz:(i+1)*sz] = sp.linalg.orth(temp)

    return theano.shared(value = np.cast[theano.config.floatX](W), name=W_name)


'''Initialize the bias'''
def initialize_bias(n, b_name):

    return theano.shared(value = np.cast[theano.config.floatX](np.zeros((n,)), \
                dtype=theano.config.floatX), name=b_name)



def histo_collapse( W_h_z, KL_Q_P_Z):

    K,M = W_h_z.shape
    len_W = np.sqrt(np.sum(W_h_z**2, axis=1))
    index = np.arange(K)

    #plt.figure(figsize=(200,100), dpi=300)# dpi=2400)
    plt.figure()# dpi=2400)
    plt.subplot(121)
    bar_width = 0.35
    opacity   = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, len_W, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='W Norm')

    plt.xlabel('Z')
    plt.ylabel('Weight Norms')
    plt.legend()

    plt.subplot(122)
    rects2 = plt.bar(index+ bar_width, KL_Q_P_Z, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='KL divergence')

    plt.xlabel('Z')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figs/bar_W_length', bbox_inches='tight')


def get_mean_var_4_latent(mean, log_var):

    mean    = mean.eval()
    log_var = log_var.eval()

    mean_per_z      = np.mean(mean, axis=0)
    log_var_per_z   = np.mean(log_var, axis=0)

    _,K = mean.shape
    index = np.arange(K)

    plt.figure()# dpi=2400)
    plt.subplot(121)
    bar_width = 0.35
    opacity   = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, mean_per_z, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='mean')

    plt.xlabel('Z')
    plt.ylabel('Mean of Gaussian per dim')
    plt.legend()

    plt.subplot(122)
    rects2 = plt.bar(index+ bar_width, log_var_per_z, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='log_var')

    plt.xlabel('Z')
    plt.ylabel('log_var of Gaussian per dim')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figs/latent_N_mean_var', bbox_inches='tight')




def get_KL_Q_P_per_Z(vae, X):

    H       = vae.get_hidden_enc(X)
    mean    = T.dot(H, vae.W_Qmean) +vae.Qm_bias
    log_var = 0.5*(T.dot(H, vae.W_Qvar) + vae.Qv_bias)

    KL_per_Z = -0.5* T.sum(1 + 2*log_var - mean**2 - T.exp(2*log_var), axis=0)

    return KL_per_Z, mean, log_var


def analysis(BNH, W_h_z, directory=None, fname=None):

    '''Covariance Heat Map'''
    covH = np.dot(BNH.T, BNH)

    plt.figure()
    plt.subplot(131)

    '''Singular values of the latent variables'''
    U,s,V = np.linalg.svd(covH)
    sortedS = np.sort(s)
    K = sortedS.shape[0]
    index = np.arange(K)

    bar_width = 1.
    opacity   = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, sortedS, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config)
    plt.xlabel('Z')
    plt.ylabel('Singular Values')
    plt.legend()

    '''Column Norm of the decoder weights'''
    K,M = W_h_z.shape
    len_W = np.sqrt(np.sum(W_h_z**2, axis=1))
    plt.subplot(132)
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, len_W, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='W Norm')

    plt.xlabel('Z')
    plt.ylabel('Weight Norms')
    plt.legend()

    plt.subplot(133)
    plt.imshow(covH)
    if fname:
        plt.savefig('./figs/analysis'+fname, bbox_inches='tight')
    else:
        plt.savefig('./figs/analysis', bbox_inches='tight')



