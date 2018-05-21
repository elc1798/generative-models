"""CS446 2018 Spring MP10.
   Implementation of a variational autoencoder for image generation.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from vaes.vae import VariationalAutoencoder
import input_data
from anime_data import *
import time

def main(_):
    """High level pipeline.

    This script performs the training for VAEs.
    """
    # Get dataset.
    # dataset = input_data.read_data_sets('MNIST_data', one_hot=True).train
    dataset = get_dataset(low_memory=False, mode="L")

    # Build model.
    NLATENT = 100
    # model = VariationalAutoencoder(ndims=28*28, nlatent=NLATENT)
    model = VariationalAutoencoder(ndims=64*64, nlatent=NLATENT)

    # Start training
    print("Training...")
    start_t = time.time()
    model.train(dataset, num_steps=5000)
    print("Done! %r", time.time() - start_t)

if __name__ == "__main__":
    tf.app.run()
