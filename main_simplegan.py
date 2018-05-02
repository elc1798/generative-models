from __future__ import print_function

import os.path
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.gan import Gan
import input_data
from anime_data import *

def run_mnist():
    mnist_dataset = input_data.read_data_sets("MNIST_data", one_hot=True)
    model = Gan(ndims=784, cdim=1, nlatent=100)
    model.train(mnist_dataset.train, img_size=28, num_steps=15000, d_steps=1)

def run_anime():
    anime_dataset = get_dataset()
    model = Gan(ndims=64*64, cdim=3, nlatent=100)
    model.train(anime_dataset, img_size=64, num_steps=50000, d_steps=1)

def main(_):
    # run_mnist()
    run_anime()

if __name__ == "__main__":
    tf.app.run()
