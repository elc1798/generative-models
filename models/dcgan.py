from __future__ import print_function

import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

from dcgenerator import Generator
from dcdiscriminator import Discriminator

class DCGAN(object):
    def __init__(self, batch_size, in_shape, g_depths, d_depths, colorspace_dim=3,
            s_size=4, n_latent=100, eta=0.0002, beta1=0.5):
        self.batch_size = batch_size
        self.s_size = s_size
        self.n_latent = n_latent # Also known as z-dim
        self.in_shape = in_shape
        self.colorspace_dim = colorspace_dim

        # Input image placeholder
        self.x_placeholder = tf.placeholder(tf.float32, [None, in_shape[0],
            in_shape[1], colorspace_dim])
        # Latent probability distribution sample placeholder
        self.z_placeholder = tf.placeholder(tf.float32, [None, 1, 1, n_latent])

        # Generator, given sample from latent space
        self.x_hat = Generator(self.z_placeholder, g_depths,
                colorspace_dim=colorspace_dim, s_size=s_size, reuse=False,
                training=True).model[-1]
        # Discriminator for fake image
        self.y_hat = Discriminator(self.x_hat, d_depths, batch_size,
                colorspace_dim=colorspace_dim, reuse=False,
                training=True).model[-1]
        # Discriminator for real
        self.y = Discriminator(self.x_placeholder, d_depths, batch_size,
                colorspace_dim=colorspace_dim, reuse=True,
                training=True).model[-1]

        # Loss tensors
        self.d_loss = Discriminator.loss(self.y, self.y_hat)
        self.g_loss = Generator.loss(self.y_hat)

        # Variable list
        self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="generator")
        self.dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="discriminator")

        print("Generator variables:")
        print(self.gen_vars)
        print("Discriminator variables:")
        print(self.dis_vars)

        # Optimizers
        self.d_optimizer = tf.train.AdamOptimizer(eta,
                beta1=beta1).minimize(self.d_loss, var_list=self.dis_vars)
        self.g_optimizer = tf.train.AdamOptimizer(eta,
                beta1=beta1).minimize(self.g_loss, var_list=self.gen_vars)

        # Session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

    def sample(self, z_mu):
        return self.session.run(self.x_hat, feed_dict={
            self.z_placeholder: z_mu
        })

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
                                tuple(list(self.in_shape) + [ self.colorspace_dim ]) )
        plt.imsave(img_dir + str(step) + ".png", out, cmap="hsv")

    def train(self, dataset, img_size=64, num_steps=5000, d_steps=1, save=True):
        z_mus = [np.random.uniform(-1,1,[1,1,1,self.n_latent]) for _ in range(400)]

        for step in range(num_steps):
            batch_x, _ = dataset.next_batch(self.batch_size)
            batch_x = batch_x.reshape((-1, img_size, img_size, self.colorspace_dim))
            batch_x = tf.image.resize_images(batch_x, list(self.in_shape)).eval()
            batch_x = (batch_x - 0.5) / 0.5
            for _ in range(d_steps):
                batch_z = np.random.uniform(-1, 1, [self.batch_size,
                    1, 1, self.n_latent])
                self.session.run(self.d_optimizer, feed_dict={
                    self.x_placeholder: batch_x,
                    self.z_placeholder: batch_z,
                })
            batch_z = np.random.uniform(-1, 1, [self.batch_size, 1, 1,
                self.n_latent])
            self.session.run(self.g_optimizer, feed_dict={
                self.z_placeholder: batch_z,
            })

            if step % 10 == 0:
                print("Step: %r of %r" % (step, num_steps))
            if step % 500 == 0:
                self.generate_file(z_mus, step)
            if save and step % 1000 == 0:
                with open("checkpoints/animeds-step_%r.pyobj" % (step,), 'w') as _f:
                    pickle.dump(dataset, _f)
                print("Saved to:", self.saver.save(self.session, "checkpoints/step_%r.ckpt" % (step,)))

