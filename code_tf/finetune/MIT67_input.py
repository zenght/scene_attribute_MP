"""Routine for decoding the MIT67 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

import tensorflow as tf

# Global constants describing the MIT67 data set.
NUM_CLASSES = 67
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5360
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1340


def get_im_list(source_dir, file_path):

    im_list = []
    im_labels = []
    with open(file_path, 'r') as fi:
        for line in fi:
            items = line.split()
            im_list.append(os.path.join(source_dir, items[0]))
            im_labels.append(int(items[1]))

    return im_list, im_labels


######
# TF dataset API
######
def tfds_input(source_dir, data_type, file_path, batch_size, new_size, MEAN_VALUE=None, norm_value=None):

    if MEAN_VALUE is None and norm_value is None:
        raise ValueError
    if MEAN_VALUE is not None and norm_value is not None:
        raise ValueError

    im_list, im_label = get_im_list(source_dir, file_path)

    height, width = new_size

    def _read_eval_function(impath, label):
        im_f = tf.read_file(impath)
        oim = tf.image.decode_jpeg(im_f, channels=3)
        rim = tf.image.resize_images(oim, [256, 256])
        # convert RGB to BGR
        # rim = tf.cast(tf.reverse(rim, axis=[-1]), tf.float32)
        if MEAN_VALUE is not None:
            mean_image = tf.convert_to_tensor(
                np.tile(MEAN_VALUE, [256, 256, 1]), tf.float32)
            rim = tf.subtract(rim, mean_image)
            print('mean norm')
        elif norm_value is not None:
            rim = rim / 255.0
            mean_image = tf.convert_to_tensor(
                np.tile(norm_value[0], [256, 256, 1]), tf.float32)
            rim = tf.subtract(rim, mean_image)
            rim /= norm_value[1]
            print('standard norm')
        if new_size == (224,224):
            # rim = tf.image.resize_image_with_crop_or_pad(rim, height, width)
            rim = tf.image.resize_images(rim, [height, width])
        elif new_size == (448,448):
            rim = tf.image.resize_images(rim,[height,width])
        elif new_size == (320,320):
            rim = tf.image.resize_images(rim, [height, width])
        print(rim.get_shape())
        return rim, label

    dataset = tf.data.Dataset.from_tensor_slices((im_list, im_label))

    print('Import the images')

        # The returned iterator will be in an uninitialized state, and you must run
        # the iterator.initializer operation before using it

    dataset = dataset.map(_read_eval_function)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element
