"""
Simple implementation of a variational autoencoder. Encoder and decoders are
simple fully connected layers.
"""

import numpy as np
import tensorflow as tf

class Simple:
    def __init__(self, img_dims, cdim=1, zdim=100):
        """
        Initializes a VAE.

        img_dims:   tuple of (image width, image height)
        cdim:       dimension of colorspace
        zdim:       dimension of latent space
        """

        assert(type(img_dims) in [list, tuple])
        assert(len(img_dims) == 2)

        self.img_dims = img_dims
        self.cdim = cdim
        self.zdim = zdim

        self.ndims = img_dims[0] * img_dims[1]

        # Create placeholders
        self.x_placeholder = tf.placeholder(tf.float32, [None, self.ndims])
        self.eta_placeholder = tf.placeholder(tf.float32, [])

        # Build the graph
        self.z_mean, self.z_log_var = self.build_encoder(self.x_placeholder)
        self.z = self.build_sampler(self.z_mean, self.z_log_var)
        self.f = self.build_decoder(self.z)

        # Loss functions
        self.loss = self.loss_total(self.f, self.x_placeholder, self.z_mean,
                self.z_log_var)
        self.optimizer = tf.train.AdamOptimizer(self.eta_placeholder).minimize(self.loss)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def build_encoder(self, x, scope="encoder", reuse=False):
        """
        Encoder segment of VAE.

        Simple implementation implements a fully connected network from:
                                         |-> z_mean
        input --> 100 nodes --> 50 nodes -
                                         |-> z_log_var

        The network utilizes tf.nn.softplus for hidden layer activations:
        https://arxiv.org/pdf/1602.02282.pdf
        """
        with tf.variable_scope(scope, reuse=reuse):
            h0 = tf.layers.dense(x, 100, activation=tf.nn.softplus,
                    name="hidden_0")
            h1 = tf.layers.dense(h0, 50, activation=tf.nn.softplus,
                    name="hidden_1")

            z_mean = tf.layers.dense(h1, self.zdim, activation=None,
                    name="z_mean")
            z_log_var = tf.layers.dense(h1, self.zdim, activation=None,
                    name="z_log_var")

            return z_mean, z_log_var

    def build_decoder(self, z, scope="decoder", reuse=False):
        """
        Decoder segment of VAE.

        Simple implementation implements a fully connected network from:
        z (zdims) --> 50 --> 100 --> ndims.

        The network utilizes tf.nn.softplus for hidden layer activations:
        https://arxiv.org/pdf/1602.02282.pdf
        """
        with tf.variable_scope(scope, reuse=reuse):
            h0 = tf.layers.dense(z, 50, activation=tf.nn.softplus,
                    name="hidden_0")
            h1 = tf.layers.dense(h0, 100, activation=tf.nn.softplus,
                    name="hidden_1")
            f = tf.layers.dense(h1, self.ndims, activation=tf.nn.sigmoid,
                    name="decoded")
            return f

    def build_sampler(self, z_mean, z_log_var, scope="sampler", reuse=False):
        """
        Samples z from latent space (determined by z_mean and z_log_var) via the
        reparametrization trick.
        """
        with tf.variable_scope(scope, reuse=reuse):
            std_dev = tf.sqrt(tf.exp(z_log_var))
            # Shape of random_normal should be (batch_size, zdim). Note that:
            # (batch_size, zdim) = (batch_size, ndim) + (0, zdim-ndim)
            sample_shape = tf.shape(self.x_placeholder) + tf.constant([0, self.zdim
                - self.ndims], dtype=tf.int32)
            return z_mean + std_dev * tf.random_normal(sample_shape)

    def loss_z(self, z_mean, z_log_var):
        """
        Loss function for the latent space
        """
        # https://github.com/tflearn/tflearn/blob/master/examples/images/variational_autoencoder.py
        # Formula taken from the bottom of page 5 of
        # https://arxiv.org/pdf/1312.6114.pdf
        kl = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl = -0.5 * tf.reduce_sum(kl, axis=1)
        return tf.reduce_mean(kl)

    def loss_x(self, f, x_gt):
        """
        Loss function for reconstruction (decoding) assuming Gaussian
        distribution
        """
        # Cross Entropy Loss
        # recon_loss = -tf.reduce_sum(x_gt * tf.log(f) + (1 - x_gt) * tf.log(1 - f), axis=1)
        # MSE Loss
        recon_loss = tf.reduce_sum(tf.square(x_gt - f), axis=1)
        return tf.reduce_mean(recon_loss)

    def loss_total(self, f, x_gt, z_mean, z_log_var):
        """
        Total Loss = Latent loss + Reconstruction loss
        """
        return self.loss_z(z_mean, z_log_var) + self.loss_x(f, x_gt)

    def sample(self, z_np):
        return self.session.run(self.f, feed_dict={self.z: z_np})

    def run_optimization_step(self, batch, optargs):
        feed_dict = { k: optargs[k] for k in optargs }
        feed_dict[self.x_placeholder] = batch

        self.session.run(self.optimizer, feed_dict=feed_dict)

