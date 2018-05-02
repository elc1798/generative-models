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
    def __init__(self, files, images=None, mode="HSV"):
        self.files = files
        self.images = images
        self.mode = mode
        self.indices = np.arange(len(self.files))
        np.random.shuffle(self.indices)
        print("Dataset loaded:", len(files))
        if images is not None:
            print(images.shape)

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

        if self.images is None:
            imgs = [ scipy.misc.imresize(
                scipy.ndimage.imread(self.files[i], mode=self.mode),
                (64,64)
            ) for i in batch ]
            return np.array(imgs), new_epoch
        else:
            return self.images[ batch ], new_epoch

def get_dataset(low_memory=True, mode="HSV"):
    # subdirs = glob.glob("anime-faces/*")
    # files = reduce(lambda a,b: a+b, [ glob.glob(d+"/*") for d in subdirs ])
    # files = random.sample(files, 50000)
    files = glob.glob("anime-faces/night/*") + glob.glob("anime-faces/black_background/*")
    if low_memory:
        anime_dataset = Dataset(files)
        return anime_dataset
    else:
        images = [ scipy.misc.imresize(
            scipy.ndimage.imread(f, mode=mode),
            (64,64)
        ) for f in files ]
        anime_dataset = Dataset(files, images=np.array(images))
        return anime_dataset

# Test next_batch robustitude
if __name__ == "__main__":
    d = get_dataset()
    for i in range(30000):
        x, _ = d.next_batch(64)
        assert(x.shape[0] == 64)
