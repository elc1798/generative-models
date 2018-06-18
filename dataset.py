import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, samples, img_dims, cdim, shuffle=True, preprocess_fn=lambda x: x):
        self.img_dims = img_dims
        self.cdim = cdim
        self.shuffle = shuffle
        self.preprocess_fn = preprocess_fn

        self.samples = samples
        self.indices = None
        self._reset()

    def _reset(self):
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next_batch(self, batch_size):
        batch = self.indices[:batch_size]
        self.indices = self.indices[batch_size:]

        new_epoch = False
        if len(self.indices) == 0:
            new_epoch = True
            self._reset()
            # We may need to pad back up to get a proper batch
            if len(batch) < batch_size:
                new_part = self.indices[:batch_size - len(batch)]
                self.indices = self.indices[batch_size - len(batch):]
                batch = np.concatenate( (batch, new_part), axis=0 )

        return self.preprocess_fn(self.samples[batch]), new_epoch

def get_dataset_mnist():
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    samples = mnist.train.images

    tf.logging.set_verbosity(old_v)
    return Dataset(samples, (28, 28), 1)

