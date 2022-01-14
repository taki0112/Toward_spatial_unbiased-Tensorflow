import numpy as np
import os
import cv2

import tensorflow as tf
from glob import glob

class Image_data:

    def __init__(self, img_size, z_dim, labels_dim, dataset_path):
        self.img_size = img_size
        self.z_dim = z_dim
        self.labels_dim = labels_dim
        self.dataset_path = dataset_path


    def image_processing(self, filename):

        x = tf.io.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=3, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize(x_decode, [self.img_size, self.img_size], antialias=True, method=tf.image.ResizeMethod.BICUBIC)
        img = preprocess_fit_train_image(img)

        latent = tf.random.normal(shape=(self.z_dim,), dtype=tf.float32)
        labels = tf.random.normal((self.labels_dim,), dtype=tf.float32)

        return img, latent, labels

    def preprocess(self):

        self.train_images = glob(os.path.join(self.dataset_path, '*.png')) + glob(os.path.join(self.dataset_path, '*.jpg'))

def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images

def random_flip_left_right(images):
    s = tf.shape(images)
    mask = tf.random.uniform([1, 1, 1], 0.0, 1.0)
    mask = tf.tile(mask, [s[0], s[1], s[2]]) # [h, w, c]
    images = tf.where(mask < 0.5, images, tf.reverse(images, axis=[1]))
    return images

def preprocess_fit_train_image(images):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    images = random_flip_left_right(images)
    images = tf.transpose(images, [2, 0, 1])

    return images

def preprocess_image(images):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    images = tf.transpose(images, [2, 0, 1])

    return images

def postprocess_images(images):
    images = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
    images = tf.transpose(images, [0, 2, 3, 1])
    images = tf.cast(images, dtype=tf.dtypes.uint8)
    return images

def merge_batch_images(images, res, rows, cols):
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

def load_images(image_path, img_width, img_height, img_channel):

    # from PIL import Image
    if img_channel == 1 :
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    else :
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = cv2.resize(img, dsize=(img_width, img_height))
    img = tf.image.resize(img, [img_height, img_width], antialias=True, method=tf.image.ResizeMethod.BICUBIC)
    img = preprocess_image(img)

    if img_channel == 1 :
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
    else :
        img = np.expand_dims(img, axis=0)

    return img

def save_images(images, size, image_path):
    # size = [height, width]
    return imsave(postprocess_images(images), size, image_path)

def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def str2bool(x):
    return x.lower() in ('true')

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def filter_resolutions_featuremaps(resolutions, featuremaps, res):
    index = resolutions.index(res)
    filtered_resolutions = resolutions[:index + 1]
    filtered_featuremaps = featuremaps[:index + 1]
    return filtered_resolutions, filtered_featuremaps

def pytorch_xavier_weight_factor(gain=0.02) :

    factor = gain * gain
    mode = 'fan_avg'

    return factor, mode

def pytorch_kaiming_weight_factor(a=0.0, activation_function='relu') :

    if activation_function == 'relu' :
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu' :
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function =='tanh' :
        gain = 5.0 / 3
    else :
        gain = 1.0

    factor = gain * gain
    mode = 'fan_in'

    return factor, mode

def automatic_gpu_usage() :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def multiple_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),
                 tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

def get_batch_sizes(gpu_num) :
    from collections import OrderedDict
    # batch size for each gpu

    if gpu_num == 1:
        x = OrderedDict([(4, 256), (8, 256), (16, 128), (32, 64), (64, 32), (128, 16), (256, 8), (512, 4), (1024, 4)])

    elif gpu_num == 2 or gpu_num == 3:
        x = OrderedDict([(4, 128), (8, 128), (16, 64), (32, 32), (64, 16), (128, 8), (256, 4), (512, 4), (1024, 4)])

    elif gpu_num == 4 or gpu_num == 5 or gpu_num == 6:
        x = OrderedDict([(4, 64), (8, 64), (16, 32), (32, 16), (64, 8), (128, 4), (256, 4), (512, 4), (1024, 4)])

    elif gpu_num == 7 or gpu_num == 8 or gpu_num == 9:
        x = OrderedDict([(4, 32), (8, 32), (16, 16), (32, 8), (64, 4), (128, 4), (256, 4), (512, 4), (1024, 4)])

    else:  # >= 10
        x = OrderedDict([(4, 16), (8, 16), (16, 8), (32, 4), (64, 2), (128, 2), (256, 2), (512, 2), (1024, 2)])

    return x

def multi_gpu_loss(x, global_batch_size):
    ndim = len(x.shape)
    no_batch_axis = list(range(1, ndim))
    x = tf.reduce_mean(x, axis=no_batch_axis)
    x = tf.reduce_sum(x) / global_batch_size

    return x

class EasyDict(dict):
    from typing import Any
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]