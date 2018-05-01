from __future__ import print_function

import glob
import random
import scipy.ndimage
import numpy as np
from skimage.color import rgb2hsv

class Dataset:
    def __init__(self, images):
        self.images = images
        self.indices = np.arange(len(images))
        np.random.shuffle(self.indices)
        print("Loaded dataset of shape", self.images.shape)

    def next_batch(self, batch_size, shuffle=True):
        batch = self.indices[:batch_size]
        self.indices = self.indices[batch_size:]
        # print("Sampling indices", batch)
        new_epoch = False
        if len(self.indices) == 0:
            self.indices = np.arange(len(self.images))
            np.random.shuffle(self.indices)
            new_epoch = True
            print("New epoch. Reset dataset")
            if len(batch) < batch_size:
                new_part = self.indices[:batch_size - len(batch)]
                self.indices = self.indices[batch_size - len(batch):]
                batch = np.concatenate((batch, new_part), axis=0)
        return self.images[batch], new_epoch

def get_dataset():
    subdirs = glob.glob("anime-faces/*")
    files = reduce(lambda a,b: a+b, [ glob.glob(d+"/*") for d in subdirs ])
    files = random.sample(files, 50000)
    images = [ scipy.ndimage.imread(f, mode="HSV") for f in files ]
    anime_dataset = Dataset(np.array(images))
    return anime_dataset

# Test next_batch robustitude
if __name__ == "__main__":
    d = get_dataset()
    for i in range(30000):
        x, _ = d.next_batch(64)
        assert(x.shape[0] == 64)
