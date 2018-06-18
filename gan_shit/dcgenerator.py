from __future__ import print_function

import numpy as np
import tensorflow as tf

class Generator:
    def __init__(self, z_input, depths, colorspace_dim=3, s_size=4,
            reuse=False, training=False):
        self.depths = depths + [ colorspace_dim ]
        self.s_size = s_size
        self.reuse = reuse
        self.training = training
        self.model = self._model(z_input)

    def _model(self, z_input):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, n_latent).
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, image_size).
        """
        model = [ z_input ]
        with tf.variable_scope("generator", reuse=self.reuse):
            with tf.variable_scope('reshape'):
                # Apply dense layer
                model.append(tf.layers.dense(
                    model[-1],
                    self.depths[0] * self.s_size * self.s_size
                ))
                # Reshape into convolvable tensor described in DCGAN paper
                model.append(tf.reshape(model[-1], [
                    -1, self.s_size, self.s_size, self.depths[0]
                ]))
                # Hit with ReLU+BatchNorm
                model.append(tf.nn.relu(tf.layers.batch_normalization(
                        model[-1],training=self.training), name='output'))
            for i in range(1, len(self.depths) - 1):
                with tf.variable_scope("deconv_%d" % (i,)):
                    # Apply deconv
                    model.append(tf.layers.conv2d_transpose(
                        model[-1],
                        self.depths[i],
                        [4, 4],
                        strides=(2, 2),
                        padding='SAME'
                    ))
                    # Hit with ReLU+BatchNorm
                    model.append(tf.nn.relu(tf.layers.batch_normalization(
                        model[-1], training=self.training), name='output'))
            with tf.variable_scope("output_layer"):
                # Apply deconv
                model.append(tf.layers.conv2d_transpose(
                    model[-1],
                    self.depths[-1],
                    [4, 4],
                    strides=(2, 2),
                    padding='SAME'
                ))
                # Hit with tanh
                model.append(tf.tanh(model[-1], name='output'))
        return model

    @staticmethod
    def loss(y_hat):
        """Loss for the generator

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

