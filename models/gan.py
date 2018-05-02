"""Generative adversarial network."""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import contrib
from tensorflow.contrib import layers

class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, batch_size=32, ndims=784, cdim=1, nlatent=150, eta_d=5e-4, eta_g=5e-4):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self.batch_size = batch_size
        self.ndims = ndims
        self.in_shape = (int(ndims**0.5), int(ndims**0.5))
        self.n_latent = nlatent
        self.colorspace_dim = cdim

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims * cdim])

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
            h1 = layers.fully_connected(x, self.ndims * self.colorspace_dim, activation_fn=tf.nn.relu)
            h2 = layers.fully_connected(h1, self.ndims * self.colorspace_dim, activation_fn=tf.nn.relu)
            y = layers.fully_connected(h2, 1, activation_fn=None, scope=scope)
            return y

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
            h1 = layers.fully_connected(z, self.ndims * self.colorspace_dim,
                    activation_fn=tf.nn.leaky_relu)
            h2 = layers.fully_connected(h1, self.ndims * self.colorspace_dim,
                    activation_fn=tf.nn.leaky_relu)
            raw = layers.fully_connected(h2, self.ndims * self.colorspace_dim,
                    activation_fn=tf.nn.tanh, scope=scope)
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

    def sample(self, z_mu):
        img = self.session.run(self.x_hat, feed_dict={
            self.z_placeholder: z_mu
        })
        # return (img+1.0) / 2.0
        return img

    def generate_file(self, z_mus, step, n=20, img_dir="images/"):
        out = np.empty( (self.in_shape[0] * n, self.in_shape[1] * n,
            self.colorspace_dim) )
        if self.colorspace_dim == 1:
            out = np.empty( (self.in_shape[0] * n, self.in_shape[1] * n) )
        for x_idx in range(n):
            for y_idx in range(n):
                img = self.sample(z_mus[x_idx + y_idx * n])
                if self.colorspace_dim == 1:
                    out[x_idx * self.in_shape[0] : (x_idx+1) * self.in_shape[0],
                        y_idx * self.in_shape[1] : (y_idx+1) * self.in_shape[1]] = img[0].reshape(self.in_shape)
                else:
                    out[x_idx * self.in_shape[0] : (x_idx+1) * self.in_shape[0],
                        y_idx * self.in_shape[1] : (y_idx+1) * self.in_shape[1]] = img[0].reshape(
                                [ self.in_shape[0], self.in_shape[1], self.colorspace_dim ] )

        plt.imsave(img_dir + str(step) + ".png", out, cmap=("hsv" if
            self.colorspace_dim == 3 else "gray"))

    def train(self, dataset, img_size=64, num_steps=5000, d_steps=1):
        z_mus = [np.random.uniform(-1,1,[1,self.n_latent]) for _ in range(400)]
        print("Generated samples")

        for step in range(num_steps):
            if step % 10 == 0:
                print("Step:", step)

            batch_x, _ = dataset.next_batch(self.batch_size)
            batch_x = batch_x.reshape((self.batch_size, self.ndims * self.colorspace_dim))
            batch_x = (batch_x - 0.5) / 0.5
            for _ in range(d_steps):
                batch_z = np.random.uniform(-1, 1, [self.batch_size,
                    self.n_latent])
                self.session.run(self.d_optimizer, feed_dict={
                    self.x_placeholder: batch_x,
                    self.z_placeholder: batch_z,
                })
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.n_latent])
            self.session.run(self.g_optimizer, feed_dict={
                self.z_placeholder: batch_z,
            })

            if step % 500 == 0:
                self.generate_file(z_mus, step)

