from __future__ import print_function

import os.path
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.dcgan import DCGAN
from anime_data import *
import input_data

def run_mnist():
    mnist_dataset = input_data.read_data_sets("MNIST_data", one_hot=True)
    model = DCGAN(32, (64, 64), [1024, 512, 256, 128], [64, 128, 256, 512],
            colorspace_dim=1)
    model.train(mnist_dataset.train, img_size=28, num_steps=15000, d_steps=1)

def run_anime():
    FNAME = "animeds.pyobj"
    anime_dataset = None
    if os.path.isfile(FNAME):
        with open(FNAME, 'r') as f:
            anime_dataset = pickle.load(f)
    else:
        anime_dataset = get_dataset()
        with open(FNAME, 'w') as f:
            pickle.dump(anime_dataset, f)

    model = DCGAN(100, (64, 64), [1024, 512, 256, 128], [64, 128, 256, 512],
            colorspace_dim=3)
    model.train(anime_dataset, img_size=96, num_steps=30000, d_steps=1)

def main(_):
    # run_mnist()
    run_anime()

if __name__ == "__main__":
    tf.app.run()
