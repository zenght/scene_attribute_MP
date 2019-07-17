
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

# Global constants describing the Attribute data set.
NUM_CLASSES = 102
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 11340
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3000
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 282925
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 75000

def get_im_list(source_dir, dest_dir, file_path):

    im_list = []
    im_labels = []
    dest_path = []
    with open(file_path, 'r') as fi:
        for line in fi:
            items = line.split()
            tp_label = []
            im_list.append(os.path.join(source_dir, items[0]))
            for i in range(1, 103):
                tp_label.append(int(items[i]))
            im_labels.append(np.array(tp_label))
            fnewname = '_'.join(items[0][:-4].split('/'))
            fdestpath = os.path.join(dest_dir, fnewname + '.npy')

            dest_path.append(fdestpath)
    print(np.array(im_labels).shape)

    # return im_list, im_labels, dest_path
    return np.array(im_list), np.array(im_labels), dest_path


######
# TF dataset API
######
def tfds_input(source_dir, data_type, file_path, batch_size, new_size,
               dest_dir='/home/haitaizeng/stanforf/DenseNet/dense_save', MEAN_VALUE=None, norm_value=None):

    if MEAN_VALUE is None and norm_value is None:
        raise ValueError
    if MEAN_VALUE is not None and norm_value is not None:
        raise ValueError

    im_list, im_labels, dest_path = get_im_list(source_dir, dest_dir, file_path)

    height, width = new_size

    def _read_train_function(impath, label, dpath):
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
        if new_size==(448,448):

            rim = tf.image.resize_images(rim,[height,width])
        elif new_size==(224,224):
            rim = tf.random_crop(rim, [height, width, 3])
            rim = tf.image.random_flip_left_right(rim)
        return rim, label, dpath

    def _read_eval_function(impath, label, dpath):
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
        if new_size==(224,224):
            rim = tf.image.resize_image_with_crop_or_pad(rim, height, width)
        elif new_size==(448,448):

            rim = tf.image.resize_images(rim,[height,width])
        return rim, label, dpath

    dataset = tf.data.Dataset.from_tensor_slices((im_list, im_labels, dest_path))
    if 'Train' in data_type:
        dataset = dataset.map(_read_train_function)
        dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        # The returned iterator will be in an uninitialized state, and you must run
        # the iterator.initializer operation before using it
    else:
        dataset = dataset.map(_read_eval_function)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
    # not call iterator.get_next() in the loop, call next_element = iterator.get_next() once
    # outside the loop, and use next_element inside the loop
    next_element = iterator.get_next()

    return iterator, next_element

