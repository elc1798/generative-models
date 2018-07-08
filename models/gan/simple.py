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

        # Build graph
        self.x_hat = self.build_generator(self.z_placeholder)
        y_hat = self.build_discriminator(self.x_hat, scope="disc_fake")
        y = self.build_discriminator(self.x_placeholder, scope="disc_real", reuse=True)

        # Loss functions
        self.loss_disc = self.loss_discriminator(y, y_hat)
        self.loss_gen = self.loss_generator(y_hat)


    def build_discriminator(self, x, scope="discriminator", reuse=False):
        """
        Discriminator for GAN. Implements a simple fully connected MLP.
        """
        with tf.variable_scope(scope, reuse=reuse) as scope:
            h1 = tf.layers.dense(x, self.ndims, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, self.ndims, activation=tf.nn.relu)
            h3 = tf.layers.dense(h2, self.ndims, activation=tf.nn.relu)
            y = tf.layers.dense(h3, self.ndims, activation=None)
            return y

    def build_generator(self, z, scope="generator", reuse=False):
        """
        Generator for GAN. From sampled z, generate image
        """
        with tf.variable_scope(scope, reuse=reuse) as scope:
            h1 = tf.layers.dense(z, self.ndims//4, activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1, self.ndims//2, activation=tf.nn.leaky_relu)
            h3 = tf.layers.dense(h2, self.ndims, activation=tf.nn.leaky_relu)
            g = tf.layers.dense(h3, self.ndims, activation=tf.nn.tanh)
            return g

    def loss_discriminator(self, y, y_hat):
        loss_y = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits_v2(
            logits = y,
            labels = tf.ones_like(y),
        ))
        loss_yhat = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits_v2(
            logits = y_hat,
            labels = tf.zeros_like(y_hat),
        ))
        return loss_y + loss_yhat

    def loss_generator(self, y_hat):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits_v2(
            logits = y_hat,
            labels = tf.ones_like(y_hat),
        ))

    def sample(self, z_np):
        pass

    def run_disc_optimization_step(self, batch, optargs):
        pass

    def run_gen_optimization_step(self, batch, optargs):
        pass
