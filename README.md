# Denoising Criterion for Variational Auto-encoding Framework 
Python (Theano) implementation of Denoising Criterion for Variational Auto-encoding Framework code provided 
by Daniel Jiwoong Im, Sungjin Ahn, Roland Memisevic, and Yoshua Bengio.
Denoising criterion injects noise in input and attempts to 
generate the original data. This is shown to be advantageous.
The codes include training criterion which corresponds to a 
tractable bound when input is corrupted. For more information, see 

```bibtex
@article{Im2016dvae,
    title={Denoising Criterion for Variational Framework},
    author={Im, Daniel Jiwoong and Ahn,Sungjin and Memisevic, Roland and Bengio, Yoshua},
    journal={http://arxiv.org/abs/1511.06406},
    year={2016}
}
```

If you use this in your research, we kindly ask that you cite the above arxiv paper


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

VAE encoder filter versus DVAE encoder filter

![Image of VAE ENC FILTER](https://raw.githubusercontent.com/jiwoongim/DVAE/master/figs/vae_enc_W)
![Image of DVAE ENC FILTER](https://raw.githubusercontent.com/jiwoongim/DVAE/master/figs/dvae_enc_W)

Here are some samples generated from trained Denoising Variational Auto-encoder (DVAE):

![Image of Freyface](https://raw.githubusercontent.com/jiwoongim/DVAE/master/figs/ff_samples.png)

Traversing over 2D latent space of trained DVAE on Freyface dataset.

![Image of Freyface](https://raw.githubusercontent.com/jiwoongim/DVAE/master/figs/ff_anal2D.png)

Here are some samples generated from trained Denoising Variational Auto-encoder (DVAE):

![Image of MNIST](https://raw.githubusercontent.com/jiwoongim/DVAE/master/figs/mnist_samples.png )


