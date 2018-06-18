from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path

import numpy as np
import matplotlib.pyplot as plt

from models.vae.simple import Simple as SimpleVAE

def imgs_to_file(imgs, pdim=(20,20), fname="output", img_dir="images"):
    """
    Generates a PNG file from samples.

    Args:
        imgs (np.ndarray): Images to draw
        pdim (tuple): (num pics per row, num pics per column)
        fname (string): File name to output as
        img_dir (string): Directory sto store file in.
    """
    assert(type(pdim) in [list, tuple] and len(pdim) == 2)
    assert(len(imgs.shape) == 4) # Should be (num samples, img width, img height, cdim)
    assert(imgs.shape[0] == pdim[0] * pdim[1])

    img_width = imgs.shape[1]
    img_height = imgs.shape[2]
    cdim = imgs.shape[3]
    out = np.empty( (img_width * pdim[0], img_height * pdim[1], cdim) )
    for x_idx in range(pdim[0]):
        for y_idx in range(pdim[1]):
            out[x_idx*img_width:(x_idx+1)*img_width,
                y_idx*img_height:(y_idx+1)*img_height] = imgs[x_idx*pdim[0]+y_idx]
    if cdim == 1:
        out = out.reshape(img_width * pdim[0], img_height * pdim[1])
    cmap = "gray" if cdim == 1 else "bgr_r"

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    plt.imsave(os.path.join(img_dir, fname), out, cmap=cmap)

def train(model, dataset, post_epoch_op, batch_size=16, N=10, optargs=dict()):
    """
    Trains the given model

    Args:
        model: A model
        dataset (Dataset): A dataset object
        post_epoch_op (func(model, iteration (int))): Called after every epoch,
            as well as before the first step and after the last step.
        batch_size (int, 16): Duh.
        N (int, 10): Number of epochs (passthroughs of the entire dataset) to
            run
    """
    is_new_epoch = True
    step = 0
    while step < N:
        if is_new_epoch:
            post_epoch_op(model, step)
            step += 1
        batch_x, is_new_epoch = dataset.next_batch(batch_size)
        model.run_optimization_step(batch_x, optargs)
    post_epoch_op(model, N)

def train_SimpleVAE(dataset, batch_size=64, zdim=100, img_dir="images"):
    model = SimpleVAE(dataset.img_dims, cdim=dataset.cdim, zdim=zdim)

    z_mus = np.random.rand( 400, zdim )
    def _post_epoch_op(m, n):
        print("Epoch %r" % (n,))
        imgs = model.sample(z_mus).reshape(400, dataset.img_dims[0],
                dataset.img_dims[1], dataset.cdim)
        imgs_to_file(imgs, pdim=(20,20), fname="step_%r" % (n,),
                img_dir=img_dir)

    train(model, dataset, _post_epoch_op, N=15, batch_size=batch_size, optargs={
        model.eta_placeholder: 5e-4,
    })

