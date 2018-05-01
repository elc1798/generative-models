from __future__ import print_function

import numpy as np
import tensorflow as tf

class Discriminator:
    def __init__(self, x_input, depths, batch_size, colorspace_dim=3,
            reuse=False, training=False):
        self.depths = [ colorspace_dim ] + depths
        self.batch_size = batch_size
        self.reuse = reuse
        self.training = training
        self.model = self._model(x_input)

    def _model(self, x_input):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, image_shape).
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1).
        """
        model = [ x_input ]
        with tf.variable_scope("discriminator", reuse=self.reuse):
            for i in range(1, len(self.depths)):
                with tf.variable_scope("conv_%d" % (i,), reuse=self.reuse):
                    # Apply deconv
                    model.append(tf.layers.conv2d(
                        model[-1],
                        self.depths[i],
                        [4, 4],
                        strides=(2, 2),
                        padding='SAME'
                    ))
                    # Hit with LeakyReLU+BatchNorm
                    model.append(tf.nn.leaky_relu(tf.layers.batch_normalization(
                        model[-1], training=self.training), name='output'))
            with tf.variable_scope("output_layer", reuse=self.reuse):
                model.append(tf.layers.conv2d(
                    model[-1],
                    self.depths[0],
                    [4, 4],
                    strides=(2, 2),
                    padding='SAME'
                ))
                model.append(tf.nn.sigmoid(model[-1], name="output"))
        return model

    @staticmethod
    def loss(y, y_hat):
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
