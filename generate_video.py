import tensorflow as tf
from PIL import Image
import numpy as np

import math
from tqdm import tqdm

import torch
from torchvision import utils
import cv2

from moviepy.editor import *

def make_grid(images, res, rows, cols):
    images = (tf.clip_by_value(images, -1.0, 1.0) + 1.0) * 127.5
    images = tf.transpose(images, perm=[0, 2, 3, 1])
    images = tf.cast(images, tf.uint8)
    images = images.numpy()

    batch_size = images.shape[0]
    assert rows * cols == batch_size
    canvas = np.zeros(shape=[res * rows, res * cols, 3], dtype=np.uint8)
    for row in range(rows):
        y_start = row * res
        for col in range(cols):
            x_start = col * res
            index = col + row * cols
            canvas[y_start:y_start + res, x_start:x_start + res, :] = images[index, :, :, :]

    return canvas

def load_generator(g_params=None, is_g_clone=True, ckpt_dir='checkpoint'):

    from networks import Generator

    if g_params is None:
        g_params = {
            'z_dim': 512,
            'w_dim': 512,
            'labels_dim': 0,
            'n_mapping': 8,
            'resolutions': [4, 8, 16, 32, 64, 128, 256],
            'featuremaps': [512, 512, 512, 512, 512, 256, 128],
            'w_ema_decay': 0.995,
            'style_mixing_prob': 0.9,
        }

    test_latent = tf.ones((1, g_params['z_dim']), dtype=tf.float32)
    test_labels = tf.ones((1, g_params['labels_dim']), dtype=tf.float32)

    # build generator model
    generator = Generator(g_params)
    _, _ = generator([test_latent, test_labels])

    if ckpt_dir is not None:
        if is_g_clone:
            ckpt = tf.train.Checkpoint(g_clone=generator)
        else:
            ckpt = tf.train.Checkpoint(generator=generator)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print(f'Generator restored from {manager.latest_checkpoint}')

    return generator

def generate():

    generator = load_generator(is_g_clone=True)
    radius = 30 # 32
    pics = 120
    truncation_psi = 0.5 # 1.0

    sample_n = 16 # 4
    n_row = 4
    n_col = 4
    res = 256
    sample_z = tf.random.normal(shape=[sample_n, 512])
    images = []
    for i in tqdm(range(pics)):
        dh = math.sin(2 * math.pi * (i / pics)) * radius
        dw = math.cos(2 * math.pi * (i / pics)) * radius

        sample_tf, _ = generator([sample_z,
                                    tf.random.normal(shape=[sample_n, 0])],
                                   shift_h=dh, shift_w=dw,
                                   training=False, truncation_psi=truncation_psi)
        # Pytorch

        sample = sample_tf
        sample = sample.numpy()
        sample = torch.Tensor(sample)
        grid = utils.make_grid(
                sample.cpu(), normalize=True, nrow=n_row, value_range=(-1, 1)
            )
        grid = grid.mul(255).permute(1, 2, 0).numpy().astype(np.uint8)
        images.append(
            grid
        )


        # Tensorflow
        # grid_tf = make_grid(sample_tf, res=res, rows=n_row, cols=n_col)
        # images.append(grid_tf)


        # Image save
        """
        for j in tqdm(range(sample_n)):
            f_name = 'images/{}_{}.png'.format(j, i)
            utils.save_image(
                sample[j].unsqueeze(0),
                f_name,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
        """

    # To video
    videodims = (images[0].shape[1], images[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"VP90")
    video = cv2.VideoWriter("sample.webm", fourcc, 24, videodims)

    for i in tqdm(images):
        video.write(cv2.cvtColor(i, cv2.COLOR_RGB2BGR))

    video.release()

    # Video to GIF
    clip = VideoFileClip("sample.webm")
    clip.write_gif("sample.gif")


generate()