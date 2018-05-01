"""Generative adversarial network."""
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers

class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, ndims=784, nlatent=150, eta_d=5e-4, eta_g=5e-4):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Add optimizers for appropriate variables
        discrim_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             "discriminator")
        print("Discriminator variables:", discrim_vars)
        self.d_optimizer = tf.train.AdamOptimizer(eta_d).minimize(self.d_loss,
                var_list=discrim_vars)
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             "generator")
        print("Generator variables:", gen_vars)
        self.g_optimizer = tf.train.AdamOptimizer(eta_g).minimize(self.g_loss,
                var_list=gen_vars)

        self.sample_gen = self.x_hat

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1).
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            h1 = layers.fully_connected(x, 784, activation_fn=tf.nn.relu)
            h2 = layers.fully_connected(h1, 784, activation_fn=tf.nn.relu)
            y = layers.fully_connected(h2, 1, activation_fn=None, scope=scope)
            return y
            """
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])

            x_image = tf.reshape(x, [-1,28,28,1])
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)

            h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*32])
            fc1 = layers.fully_connected(h_pool1_flat, 1024,
                    activation_fn=tf.nn.relu)
            fc2 = layers.fully_connected(fc1, 512,
                    activation_fn=tf.nn.relu)
            y = layers.fully_connected(fc2, 1, activation_fn=None,
                    scope=scope)
            return y
            """


    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        # Raw formula from the original paper
        # l = tf.reduce_mean(tf.log(y)) - tf.reduce_mean(tf.log(1 - y_hat))
        # return l
        # Softmax Cross Entropy With Logits formulation:
        loss_y = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y, labels=tf.ones_like(y)))
        loss_yhat = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_hat, labels=tf.zeros_like(y_hat)))
        return loss_y + loss_yhat


    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            h1 = layers.fully_connected(z, 784, activation_fn=tf.nn.leaky_relu)
            h2 = layers.fully_connected(h1, 784, activation_fn=tf.nn.leaky_relu)
            raw = layers.fully_connected(h2, 784, activation_fn=tf.nn.leaky_relu, scope=scope)
            return raw


    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        # Raw formula from paper
        # return tf.reduce_mean(tf.log(1 - y_hat))
        # Paper's suggestion on improvement
        # return -tf.reduce_mean(tf.log(y_hat))
        # Softmax Cross Entropy With Logits formulation:
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
           logits=y_hat, labels=tf.ones_like(y_hat)))
