#coding:utf-8
"""
A binary to finetune pre-trained DenseNet161 model on MIT67
Author: Haitao Zeng
"""
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import numpy as np
from datetime import datetime
import time
import yaml
from easydict import EasyDict as edict

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
from easydict import EasyDict as edict
import MIT67_input
from DenseNet import DenseNet



# Global constants describing the MIT67 data set.
IMAGE_SIZE = DenseNet.IMAGE_SIZE
NUM_CLASSES = MIT67_input.NUM_CLASSES
# IMAGENET_MEAN with BGR format
IMAGENET_MEAN = [104.00698793, 116.66876762, 122.67891434]
PLACES365_MEAN = [104.05100722, 112.51448911, 116.67603893]
PLACES205_MEAN = [104.00646973, 116.66902161, 122.67946625]
# standard normalization (mean, std)
PLACES365_Norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

allskip_layers = {'conv0': [],
                'DB1': ['conv0', 'norm0'],
                'DB2': ['conv0', 'norm0', 'denseblock1', 'transition1'],
                'DB3': ['conv0', 'norm0', 'denseblock1', 'transition1',
                       'denseblock2', 'transition2'],
                'DB4': ['conv0', 'norm0', 'denseblock1', 'transition1',
                       'denseblock2', 'transition2', 'denseblock3',
                       'transition3'],
                'fc5': ['conv0', 'norm0', 'denseblock1', 'transition1',
                       'denseblock2', 'transition2', 'denseblock3',
                       'transition3', 'denseblock4', 'norm5']}

preload_layers = ['conv0', 'norm0', 'denseblock1', 'transition1',
                  'denseblock2', 'transition2', 'denseblock3',
                  'transition3', 'denseblock4', 'norm5']
display_step = 60



def readconfig(config_path):
    with open(config_path, 'r') as f:
        CONFIG = edict(yaml.load(f))
    return CONFIG

def load_init_ckpt(path, session):
    not_load_layers = []

    if path == 'None':
        raise ValueError('Please supply the path to a checkpoint of model')

    print('Loading the weights of {}'.format(path))
    cp_vars = tf.contrib.framework.list_variables(path)
    load_layers = {}
    tf.get_variable_scope().reuse_variables()
    for var_name, _ in cp_vars:
        tmp_layer = var_name.split('/')[0]
        if tmp_layer not in not_load_layers:

            # tf.get_variable_scope().reuse_variables()
            try:

                load_layers[var_name] = tf.get_variable(var_name)
            except:
                # if tmp_layer =='fc5':
                #     print('Not Load Layer: {}'.format(var_name))

                print('Not Load Layer: {}'.format(var_name))
                continue

    print('----------Alreadly loaded variables--------')
    # for k in load_layers:
    #     print(k)


    init_fn=tf.contrib.framework.assign_from_checkpoint_fn(path,
                                                    load_layers,
                                                    ignore_missing_vars=True,
                                                    reshape_variables=True)
    init_fn(session)

    print('Loading the weights is Done.')

class model_flag():

    def __init__(self):
        self.dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path =  os.path.join(self.dir_path, "configs/MIT67.yaml")
        self.CONFIG = readconfig(self.config_path)
        self.train_dir = os.path.join(self.dir_path, "data/MIT67_model")
        self.im_dir = self.CONFIG.DATASET.ROOT
        self.num_epochs = 40
        self.batch_size = 16
        self.net_depth = 161
        self.nlayers = [6, 12, 36, 24]
        self.growth_rate = 48
        self.bn_size = 4
        self.compressed = 0.5
        self.num_classes = self.CONFIG.DATASET.N_CLASSES
        self.moving_average_decay = 0.99
        self.num_epochs_per_decay = 10
        self.learning_rate_decay_factor = 0.1
        self.initial_learning_rate = 0.001
        self.weights_path = self.CONFIG.PRETRAINED.WEIGHT_PATH
        self.MEAN_VALUE = None
        self.skip_layers = None
        self.wBNm = True  # whether to init BN moving_* with pre-trained para


FLAGS = model_flag()


def inputs(data_type, file_path):
    if not FLAGS.im_dir:
        raise ValueError('Please supply a image dir')
    data_iter, data = MIT67_input.tfds_input(source_dir=FLAGS.im_dir,
                                        data_type=data_type,
                                        file_path=file_path,
                                        batch_size=FLAGS.batch_size,
                                        new_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        MEAN_VALUE=None,
                                        norm_value=PLACES365_Norm)

    return data_iter, data


def finetune(init_lr, ftlayer):
    """finetune pre-trained DenseNet161 model on MIT67 dataset."""
    tf.reset_default_graph()
    FLAGS.initial_learning_rate = init_lr
    ckptstr = '-{}'.format(ftlayer)
    skip_layers = allskip_layers[ftlayer]

    if not FLAGS.wBNm:
        ckptstr += '-woBNm0.9'
    else:
        ckptstr += '-wBNm0.9'
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(dir_path, "configs/MIT67.yaml")
    CONFIG = readconfig(config_path)
    train_file = CONFIG.DATASET.SPLIT.TRAIN
    test_file = CONFIG.DATASET.SPLIT.TEST
    train_iter, train_data = inputs('Train', train_file)
    test_iter, test_data = inputs('Test', test_file)
    dtype = tf.float32
    x = tf.placeholder(dtype, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(dtype)
    is_training = tf.placeholder(tf.bool)
    global_step = tf.train.get_or_create_global_step()

    # Start Inference
    model = DenseNet(FLAGS)
    score = model.inference(x, is_training=is_training, keep_prob=keep_prob)
    loss = model.loss(score, y)
    train_op = model.train(loss, MIT67_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
                           preload_layers, skip_layers, global_step)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("Accuracy"):
        correct_pred = tf.equal(tf.argmax(score, 1),
                                tf.cast(y, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype))
        tf.summary.scalar("Accuarcy", accuracy)

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    summaryfiles_path = os.path.join(FLAGS.train_dir, 'summaryfiles')
    checkpoints_path = os.path.join(FLAGS.train_dir, 'checkpoints')
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()
    # Initialize the FileWriter

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Configuration of GPU usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    pretrained = CONFIG.PRETRAINED.NAME
    # Start Tensorflow session
    with tf.Session(config=config) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # Add the model graph to TensorBoard
        train_writer = tf.summary.FileWriter(summaryfiles_path + "/train", sess.graph)
        test_writer = tf.summary.FileWriter(summaryfiles_path + '/test')
        # Load the pretrained weights into the non-trainable layers
        # needs 250~400s
        # Attribute Model
        if pretrained not in ['IMAGENET', ' PLACES365']:
            load_model = CONFIG.PRETRAINED.WEIGHT_PATH
            load_init_ckpt(load_model, sess)

        else:
            ## Petrained Places365 or IMAGENET
            model.load_initial_weights(sess, FLAGS.wBNm)

        print("{} start training...".format(datetime.now()))

        # Loop over number of epochs
        prev_acc = []
        for epoch in range(FLAGS.num_epochs):
            # Train the model
            epoch_sta = '*' * 15 + ' Epoch number: {} ' + '*' * 15
            sess.run(train_iter.initializer)
            print(epoch_sta.format(epoch + 1))
            step = 1
            while True:
                try:
                    if step % display_step == 1:
                        start_time = time.time()
                        ds_acc = []
                        ds_loss = []
                    # run the training op
                    batch_im, batch_label = sess.run(train_data)
                    merged, los, acc, _, lr = sess.run([merged_summary, loss, accuracy, train_op, model.lr],
                                                       feed_dict={x: batch_im,
                                                                  y: batch_label,
                                                                  keep_prob: 1.0,
                                                                  is_training: True})
                    ds_acc.append(acc)
                    ds_loss.append(los)
                    train_writer.add_summary(merged, epoch)
                    if step % display_step == 0:
                        dsm_acc = np.mean(ds_acc, dtype=np.float32)
                        dsm_loss = np.mean(ds_loss, dtype=np.float32)
                        out_str = "Step number: {} | time: {:.4f} | learning_rate: {}\n" + \
                                  "Loss: \033[1;31;40m{:.4f}\033[0m, Accuracy: \033[1;31;40m{:.4f}\033[0m"
                        print(out_str.format(step, time.time() - start_time, lr,
                                             float(dsm_loss), float(dsm_acc)))

                    step += 1

                except tf.errors.OutOfRangeError:
                    break

            # Evaluate the model on entire validation set
            print("{} Strat Evaluating".format(datetime.now()))
            sess.run(test_iter.initializer)
            eval_acc = 0.0
            eval_loss = 0.0
            eval_count = 0
            vstart_time = time.time()
            while True:
                try:
                    batch_im, batch_label, _ = sess.run(test_data)
                    bnum = len(batch_im)

                    merged, vacc, vlos = sess.run([merged_summary, accuracy, loss],
                                                  feed_dict={x: batch_im,
                                                             y: batch_label,
                                                             keep_prob: 1.0,
                                                             is_training: False})
                    eval_acc += vacc * bnum
                    eval_loss += vlos * bnum
                    eval_count += bnum
                    test_writer.add_summary(merged, epoch)
                except tf.errors.OutOfRangeError:
                    break
            print('Actual Test Number: {}'.format(eval_count))
            eval_acc /= eval_count
            eval_loss /= eval_count
            vout_str = "time: {:.4f} | Loss = \033[1;34;48m{:.4f}\033[0m | Accuracy = \033[1;34;48m{:.4f}\033[0m\n"
            print(vout_str.format(time.time() - vstart_time, eval_loss, eval_acc))

            if len(prev_acc) > 1 and eval_acc > max(prev_acc):
                # Save checkpoint of the model
                print("{} Saving checkpoint of model...".format(datetime.now()))

                checkpoint_name = os.path.join(checkpoints_path,
                                               '{}-DenseNet161{}-Momentum{}_epoch{}_{:.4f}.ckpt'.format(pretrained,
                                                   ckptstr, FLAGS.initial_learning_rate, epoch + 1, eval_acc))
                save_path = saver.save(sess, checkpoint_name, write_meta_graph=False)

                print("{} Model checkpoint saved at {}".format(datetime.now(), save_path))

            prev_acc.append(eval_acc)

        print("Done training -- {} epochs limit reached".format(FLAGS.num_epochs))



def main(argv=None):
    lrs = [0.001]
    # batch_size-32 ftlayer< DB1, 24 ftlayer = DB1, 16-conv0
    ftls = ['DB1']
    for lr in lrs:
        for ftl in ftls:
            finetune(lr, ftl)


if __name__ == '__main__':
    tf.app.run()
