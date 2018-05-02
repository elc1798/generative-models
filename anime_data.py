from __future__ import print_function

import glob
import random
import scipy.ndimage
import scipy.misc
import numpy as np
from skimage.color import rgb2hsv

# The dataset is extremely large... it's better to read the images on the fly
# rather than caching like... 8 gb worth of photos...

class Dataset:
    def __init__(self, files):
        self.files = files
        self.indices = np.arange(len(self.files))
        np.random.shuffle(self.indices)

    def next_batch(self, batch_size, shuffle=True):
        batch = self.indices[:batch_size]
        self.indices = self.indices[batch_size:]
        # print("Sampling indices", batch)
        new_epoch = False
        if len(self.indices) == 0:
            self.indices = np.arange(len(self.files))
            np.random.shuffle(self.indices)
            new_epoch = True
            print("New epoch. Reset dataset")
            if len(batch) < batch_size:
                new_part = self.indices[:batch_size - len(batch)]
                self.indices = self.indices[batch_size - len(batch):]
                batch = np.concatenate((batch, new_part), axis=0)

        imgs = [ scipy.misc.imresize(
            scipy.ndimage.imread(self.files[i], mode="HSV"),
            (64,64)
        ) for i in batch ]
        return np.array(imgs), new_epoch

def get_dataset():
    subdirs = glob.glob("anime-faces/*")
    files = reduce(lambda a,b: a+b, [ glob.glob(d+"/*") for d in subdirs ])
    files = random.sample(files, 50000)
    anime_dataset = Dataset(files)
    return anime_dataset

# Test next_batch robustitude
if __name__ == "__main__":
    d = get_dataset()
    for i in range(30000):
        x, _ = d.next_batch(64)
        assert(x.shape[0] == 64)
