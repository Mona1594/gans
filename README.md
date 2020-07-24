# GANs

GANs or Generative Adversarial Networks is the one of the most interesting things you can work in ML field.  They propose in 2014 in the paper called [Generative Adversarial Nets][gans] by Ian Goodfellow et al. These model solves a *unsupervised learning problem* as opposed to [Variational Autoencoders][vae] which solves *semi-supervised learning problem*.

But these models have their own problems as well. Generative Adversarial Models are *very hard to train* and often *mode collapse* (generating same output for different inputs) is observed. Despite of these issues, these models managed to produce high quality images of human faces and can transform images from one domain to another.

[gans]: https://arxiv.org/abs/1406.2661
[vae]: https://papers.nips.cc/paper/6528-variational-autoencoder-for-deep-learning-of-images-labels-and-captions.pdf


## Deep Convolutional GAN

Uses the classical GAN log loss and uses Convolutional layer instead of linear layers to produce images.
`Conv2DTranspose` is used to upsample images.

First introduced in the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks][dcgan-paper] by Alec Radford et al.<br>

[🔗Code implementation][dcgan-code]<br>

<p align="center">
  <img src="https://github.com/joshiprashanthd/gans/blob/master/tensorflow-models/gifs/dcgan.gif" />
</p>
<br>

[dcgan-code]: https://github.com/joshiprashanthd/gans/blob/master/tensorflow-models/dcgan.py
[dcgan-paper]: https://arxiv.org/abs/1511.06434

## Conditional GAN
These models are used to produce images based on some prior condition. These prevent the models to randomly generate images.

Introduced in paper [Condtional Generative Adversarial Nets][cond-paper] by the authors Mehdi Mirza, Simon Osindero.<br>

[🔗Code implementation][cond-code]<br>

<p align="center">
  <img src="https://github.com/joshiprashanthd/gans/blob/master/tensorflow-models/gifs/conditional_gan_image.gif" />
</p>
<br>

[cond-paper]: https://arxiv.org/abs/1411.1784
[cond-code]: https://github.com/joshiprashanthd/gans/blob/master/tensorflow-models/conditional_gan.py

## Wasserstein GAN
These models provides more stabilization of training over the classical GANs. It uses Wasseretein Loss which difference between
mean of fake predictions and real predictions. Since it does not use expectation in the loss, we do not use sigmoid on discriminator output.
It also apply weight clipping on the weights of conv layers.

Introduced in paper [Wasserstein GAN][wgan-paper] by Martin Arjovsky et al.

[🔗Code implementation][wgan-code]<br>

<p align="center">
  <img src="https://github.com/joshiprashanthd/gans/blob/master/tensorflow-models/gifs/improved-wgan.gif" />
</p>
<br>

[wgan-paper]: https://arxiv.org/abs/1701.07875
[wgan-code]: https://github.com/joshiprashanthd/gans/blob/master/tensorflow-models/wassertein_gan.py

## Improved Wasserstein GAN
These models a adds new term, *a penalty term*, which penalizes the discriminator loss. It does not use weight clipping.

Introduced in paper [Improved Training of Wasserstein GAN][impr-wgan-paper] in 2017 by Ishaan Gulrajani et al.

[🔗Code implementation][impr-wgan-code]<br>

<p align="center">
  <img src="https://github.com/joshiprashanthd/gans/blob/master/tensorflow-models/gifs/improved-wgan-cifar10.gif" />
</p>
<br>

[impr-wgan-paper]: https://arxiv.org/abs/1704.00028
[impr-wgan-code]: https://github.com/joshiprashanthd/gans/blob/master/tensorflow-models/improved_wassertein_gan.py
