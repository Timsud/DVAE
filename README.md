# Denoising Criterion for Variational Auto-encoding Framework 


Python (Theano) implementation of Denoising Criterion for Variational Auto-encoding Framework code provided 
by Daniel Jiwoong Im, Sungjin Ahn, Roland Memisevic, and Yoshua Bengio.
The codes include experiments on hodge decomposition, in particular convservative components (for now),
and vector field deformations in 2D. For more information, see 

```bibtex
@article{Im2016dvae,
    title={Denoising Criterion for Variational Framework},
    author={Im, Daniel Jiwoong and Ahn,Sungjin and Memisevic, Roland and Bengio, Yoshua},
    journal={http://arxiv.org/abs/1511.06406},
    year={2016}
}
```

If you use this in your research, we kindly ask that you cite the above workshop paper


## Dependencies
Packages
* [numpy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/)


## How to run
Entry code for one-bit flip and factored minimum probability flow for mnist data are 
```
    - /main_mnist.py
    - /main_freyface.py
```

Here are some samples generated from trained Denoising Variational Auto-encoder (DVAE):

![Image of Freyface](https://raw.githubusercontent.com/jiwoongim/DVAE/master/figs/ff_samples.png)

Traversing over 2D latent space of trained DVAE on Freyface dataset.

![Image of Freyface](https://raw.githubusercontent.com/jiwoongim/DVAE/master/figs/ff_anal2D.png)

Here are some samples generated from trained Denoising Variational Auto-encoder (DVAE):

![Image of MNIST](https://raw.githubusercontent.com/jiwoongim/DVAE/master/figs/mnist_samples.png )


