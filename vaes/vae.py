"""Variation autoencoder."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import contrib
from tensorflow.contrib import layers
from tensorflow.contrib.slim import fully_connected


class VariationalAutoencoder(object):
    """Varational Autoencoder.
    """
    def __init__(self, ndims=784, cdim=1, nlatent=10):
        """Initializes a VAE. (**Do not change this function**)

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent
        self.colorspace_dim = cdim
        self.in_shape = (int(ndims**0.5), int(ndims**0.5))

        # Create session
        self.session = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        self.z_mean, self.z_log_var = self._encoder(self.x_placeholder)
        self.z = self._sample_z(self.z_mean, self.z_log_var)
        self.outputs_tensor = self._decoder(self.z)

        # Setup loss tensor, predict_tensor, update_op_tensor
        self.loss_tensor = self.loss(self.outputs_tensor, self.x_placeholder,
                                     self.z_mean, self.z_log_var)

        self.update_op_tensor = self.update_op(self.loss_tensor,
                                               self.learning_rate_placeholder)

        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())

    def _sample_z(self, z_mean, z_log_var):
        """Samples z using reparametrization trick.

        Args:
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, _nlatent)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, _nlatent)
        Returns:
            z (tf.Tensor): Random sampled z of dimension (None, _nlatent)
        """

        return z_mean + tf.sqrt(tf.exp(z_log_var)) * tf.random_normal(
                tf.shape(self.x_placeholder) + tf.constant([0, self._nlatent -
                    self._ndims], dtype=tf.int32))

    def _encoder(self, x):
        """Encoder block of the network.

        Builds a two layer network of fully connected layers, with 100 nodes,
        then 50 nodes, and outputs two branches each with _nlatent nodes
        representing z_mean and z_log_var. Network illustrated below:

                             |-> _nlatent (z_mean)
        Input --> 100 --> 50 -
                             |-> _nlatent (z_log_var)

        Use activation of tf.nn.softplus for hidden layers.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, _ndims).
        Returns:
            z_mean(tf.Tensor): The latent mean, tensor of dimension
                (None, _nlatent).
            z_log_var(tf.Tensor): The latent log variance, tensor of dimension
                (None, _nlatent).
        """
        ####### Implementation Here ######
        hidden1 = fully_connected(x, 100, activation_fn=tf.nn.softplus)
        hidden2 = fully_connected(hidden1, 50, activation_fn=tf.nn.softplus)
        z_mean = fully_connected(hidden2, self._nlatent, activation_fn=None)
        z_log_var = fully_connected(hidden2, self._nlatent, activation_fn=None)
        return z_mean, z_log_var

    def _decoder(self, z):
        """From a sampled z, decode back into image.

        Builds a three layer network of fully connected layers,
        with 50, 100, _ndims nodes.

        z (_nlatent) --> 50 --> 100 --> _ndims.

        Use activation of tf.nn.softplus for hidden layers.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, _nlatent).
        Returns:
            f(tf.Tensor): Decoded features, tensor of dimension (None, _ndims).
        """

        ####### Implementation Here ######
        hidden1 = fully_connected(z, 50, activation_fn=tf.nn.softplus)
        hidden2 = fully_connected(hidden1, 100, activation_fn=tf.nn.softplus)
        f = fully_connected(hidden2, self._ndims, activation_fn=tf.nn.sigmoid)
        return f

    def _latent_loss(self, z_mean, z_log_var):
        """Constructs the latent loss.

        Args:
            z_mean(tf.Tensor): Tensor of dimension (None, _nlatent)
            z_log_var(tf.Tensor): Tensor of dimension (None, _nlatent)

        Returns:
            latent_loss(tf.Tensor): A scalar Tensor of dimension ()
                containing the latent loss.
        """
        ####### Implementation Here ######
        # https://github.com/tflearn/tflearn/blob/master/examples/images/variational_autoencoder.py
        # Formula taken from the bottom of page 5 of
        # https://arxiv.org/pdf/1312.6114.pdf
        kl = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl = -0.5 * tf.reduce_sum(kl, axis=1)
        return tf.reduce_mean(kl)

    def _reconstruction_loss(self, f, x_gt):
        """Constructs the reconstruction loss, assuming Gaussian distribution.

        Args:
            f(tf.Tensor): Predicted score for each example, dimension (None,
                _ndims).
            x_gt(tf.Tensor): Ground truth for each example, dimension (None,
                _ndims).
        Returns:
            recon_loss(tf.Tensor): A scalar Tensor for dimension ()
                containing the reconstruction loss.
        """
        ####### Implementation Here ######
        # Cross Entropy Loss
        recon_loss = -tf.reduce_sum(x_gt * tf.log(f) + (1 - x_gt) * tf.log(1 - f), axis=1)
        # MSE Loss
        # recon_loss = tf.reduce_sum(tf.square(x_gt - f), axis=1)
        return tf.reduce_mean(recon_loss)

    def loss(self, f, x_gt, z_mean, z_log_var):
        """Computes the total loss.

        Computes the sum of latent and reconstruction loss.

        Args:
            f (tf.Tensor): Decoded image for each example, dimension (None,
                _ndims).
            x_gt (tf.Tensor): Ground truth for each example, dimension (None,
                _ndims)
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, _nlatent)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, _nlatent)

        Returns:
            total_loss: Tensor for dimension (). Sum of
                latent_loss and reconstruction loss.
        """
        ####### Implementation Here ######
        total_loss = self._latent_loss(z_mean, z_log_var) + \
            self._reconstruction_loss(f, x_gt)
        return total_loss

    def update_op(self, loss, learning_rate):
        """Creates the update optimizer.

        Use tf.train.AdamOptimizer to obtain the update op.

        Args:
            loss(tf.Tensor): Tensor of shape () containing the loss function.
            learning_rate(tf.Tensor): Tensor of shape (). Learning rate for
                gradient descent.
        Returns:
            train_op(tf.Operation): Update opt tensorflow operation.
        """
        ####### Implementation Here ######
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op

    def sample(self, z_np):
        """Generates random samples from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension
                (batch_size, _nlatent).

        Returns:
            out(numpy.ndarray): The sampled images (numpy.ndarray) of
                dimension (batch_size, _ndims).
        """
        ####### Implementation Here ######
        out = self.session.run(self.outputs_tensor, feed_dict={
            self.z: z_np
        })
        return out

    def generate_file(self, z_mus, step, n=20, img_dir="images/"):
        # Plot out latent space
        out = np.empty((self.in_shape[0]*20, self.in_shape[0]*20))
        for x_idx in range(20):
            for y_idx in range(20):
                z_mu = z_mus[x_idx + y_idx * 20]
                img = self.sample(z_mu)
                out[x_idx*self.in_shape[0]:(x_idx+1)*self.in_shape[0],
                    y_idx*self.in_shape[0]:(y_idx+1)*self.in_shape[0]] = img[0].reshape(self.in_shape[0], self.in_shape[0])
        plt.imsave('images/step_%r.png' % (step,), out, cmap="gray")

    def train(self, dataset, batch_size=16, img_size=64, num_steps=5000, d_steps=1):
        span = 4
        z_mus = [np.random.uniform(-span,span,[1,self._nlatent]) for _ in range(400)]
        print("Generated samples")

        for step in range(num_steps):
            if step % 10 == 0:
                print("Step:", step)

            batch_x, _ = dataset.next_batch(batch_size)
            batch_x = batch_x.reshape((batch_size, self._ndims * self.colorspace_dim))
            self.session.run(self.update_op_tensor, feed_dict={
                self.x_placeholder: batch_x,
                self.learning_rate_placeholder: 0.0005,
            })

            if step % 500 == 0:
                self.generate_file(z_mus, step)

