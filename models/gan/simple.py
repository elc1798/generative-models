"""
Simple implementation of a generative adversarial network. Uses fully connected
layers in the generator and discriminator.
"""

import numpy as np
import tensorflow as tf

class Simple:
    def __init__(self, img_dims, cdim=1, zdim=100):
        """
        Initializes a GAN.

        Args:
            img_dims:   tuple of (image width, image height)
            cdim:       dimension of colorspace
            zdim:       dimension of latent space
        """
        assert(type(img_dims) in [list, tuple])
        assert(len(img_dims) == 2)

        self.img_dims = img_dims
        self.cdim = cdim
        self.zdim = zdim

        self.ndims = img_dims[0] * img_dims[1] * cdim

        # Placeholders
        self.x_placeholder = tf.placeholder(tf.float32, [None, self.ndims])
        self.z_placeholder = tf.placeholder(tf.float32, [None, self.zdim])
        self.eta_placeholder = tf.placeholder(tf.float32,[])

    def build_discriminator(self, x, scope="discriminator", reuse=False):
        """
        Discriminator for GAN.
        """
        pass
