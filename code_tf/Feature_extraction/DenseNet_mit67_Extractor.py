"""Builds the VGG16 network."""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import yaml
from easydict import EasyDict as edict
import tensorflow as tf
from DenseNet import BatchNorm,conv,avgpool,max_pool,Relu,DenseBlock,Transition


preload_layers = ['conv0', 'norm0', 'denseblock1', 'transition1',
                  'denseblock2', 'transition2', 'denseblock3',
                  'transition3', 'denseblock4', 'norm5']



def readconfig(config_path):
    with open(config_path, 'r') as f:
        CONFIG = edict(yaml.load(f))
    return CONFIG



class DenseNet(object):

    def __init__(self, num_classes, weights_path):

        # Parse input arguments into class variables
        self.num_classes = num_classes
        self.WEIGHTS_PATH = weights_path
        self.net_depth = 161
        self.nlayers = [6, 12, 36, 24]
        self.growth_rate = 48
        self.bn_size = 4
        self.compressed = 0.5

    def inference(self, X, is_training, keep_prob):

        # First Convluation (224x224)
        x = conv(X, 7, 2 * self.growth_rate, 2, pad=3, name='conv0')
        x = BatchNorm(x, is_training, name='norm0')
        x = Relu(x)
        print(x.shape)
        x = max_pool(x, 3, 2, pad=1)
        print(x.shape)

        # DenseBlock 1 (56x56)
        x = DenseBlock(x, self.nlayers[0], self.bn_size, self.growth_rate, is_training,
                       keep_prob, name='denseblock1')
        x = Transition(x, self.compressed, is_training, keep_prob, name='transition1')
        print(x.shape)

        # DenseBlock 2 (28x28)
        x = DenseBlock(x, self.nlayers[1], self.bn_size, self.growth_rate, is_training,
                       keep_prob, name='denseblock2')
        x = Transition(x, self.compressed, is_training, keep_prob, name='transition2')
        print(x.shape)

        # DenseBlock 3 (14x14)
        x = DenseBlock(x, self.nlayers[2], self.bn_size, self.growth_rate, is_training,
                       keep_prob, name='denseblock3')
        x = Transition(x, self.compressed, is_training, keep_prob, name='transition3')
        print(x.shape)

        # DenseBlock 4 (7x7)
        x = DenseBlock(x, self.nlayers[3], self.bn_size, self.growth_rate, is_training,
                       keep_prob, name='denseblock4')
        x = BatchNorm(x, is_training, name='norm5')
        x = Relu(x)
        ox = avgpool(x, 7, 1, padding='VALID')
        print(ox.shape)
        # x = tf.squeeze(ox, [1, 2])
        # print(x.shape)

        # classifier
        x = conv(ox, 1, self.num_classes, 1, name='fc5', with_bias=True)
        x = tf.squeeze(x, [1, 2], name='fc5/squeeze')
        # x = fc(x, self.num_classes, name='fc5')

        return x

    def load_initial_weights(self, session, wBNm):

        not_load_layers = ['fc5']
        if wBNm:
            wBNm_key = ' '
        else:
            wBNm_key = 'moving'
        st = time.time()
        if self.WEIGHTS_PATH == 'None':
            raise ValueError('Please supply the path to a pre-trained model')

        print('Loading the weights of pre-trained model')

        # load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        out_layers = []
        # merge all assign ops, just run once can obtain low overhead of time.
        assign_ops = []
        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            if wBNm_key in op_name:
                continue
            layer = op_name.split('/')[0]
            # Check if the layer is one of the layers that should be reinitialized
            if layer not in not_load_layers:
                data = weights_dict[op_name]
                tf.get_variable_scope().reuse_variables()
                try:
                    # Biases
                    var = tf.get_variable(op_name)
                    assign_ops.append(var.assign(data))
                    # print("Loaded layer: {}".format(op_name))
                except:
                    print('Not Load Layer: {}'.format(op_name))
                    continue
                out_layers.append(layer)
        session.run(assign_ops)
        prt_strs = ["{} : {}".format(k, out_layers.count(k)) for k in sorted(set(out_layers))]
        for pstr in prt_strs:
            print(pstr)

        print('Loading the weights is Done in {:.4f}s.'.format(time.time() - st))


# IMAGENET_MEAN with BGR format
IMAGENET_MEAN = [104.00698793, 116.66876762, 122.67891434]
PLACES365_MEAN = [104.05100722, 112.51448911, 116.67603893]
PLACES205_MEAN = [104.00646973, 116.66902161, 122.67946625]
# standard normalization (mean, std)
PLACES365_Norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# IMAGENET_Norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # RGB format
image_size = 224
rsize = 112
rstride = 32
rnum = (int(np.floor((image_size-rsize)/float(rstride))) + 1) ** 2



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



def load_data(file_path, batch_size, CONFIG):
    im_dir = CONFIG.DATASET.ROOT
    dest_dir = CONFIG.EXP.OUTPUT_DIR
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    im_list = []
    im_labels = []
    dest_path = []
    with open(file_path, 'r') as fi:
        for line in fi:
            im_list.append(os.path.join(im_dir, line.split()[0]))
            im_labels.append(int(line.split()[-1]))
            fnewname = '_'.join(line.split()[0][:-4].split('/'))
            dest_path.append(os.path.join(dest_dir, fnewname+'.npy'))

    height = width = image_size
    MEAN_VALUE = None
    norm_value = PLACES365_Norm
    def _read_eval_function(impath, label, dpath):
        im_f = tf.read_file(impath)
        oim = tf.image.decode_jpeg(im_f, channels=3)
        rim = tf.image.resize_images(oim, [height, width])
        if MEAN_VALUE is not None:
            mean_image = tf.convert_to_tensor(
                np.tile(MEAN_VALUE, [height, width, 1]), tf.float32)
            rim = tf.subtract(rim, mean_image)
            print('mean norm')
        elif norm_value is not None:
            rim = rim / 255.0
            mean_image = tf.convert_to_tensor(
                np.tile(norm_value[0], [height, width, 1]), tf.float32)
            rim = tf.subtract(rim, mean_image)
            rim /= norm_value[1]
            print('standard norm')
        return rim, label, dpath

    dataset = tf.data.Dataset.from_tensor_slices((im_list, im_labels, dest_path))
    dataset = dataset.map(_read_eval_function)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def extract_feat(phase):

    batch_size = 20
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(dir_path, "configs/MIT67_extra.yaml")
    CONFIG = readconfig(config_path)
    if phase == 'Train':
        extract_file = CONFIG.DATASET.SPLIT.TRAIN
    elif phase == 'Test':
        extract_file = CONFIG.DATASET.SPLIT.TEST
    extract_iter, extract_data = load_data(extract_file, batch_size, CONFIG)
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    is_training = tf.placeholder(tf.bool, None)
    keep_prob = tf.placeholder(tf.float32, None)
    model = DenseNet(67, weights_path=None)
    score = model.inference(x, is_training, keep_prob)

    # Configuration of GPU usage
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True

    # Start Tensorflow session
    with tf.Session(config=config) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        ### LOAD WEIGHT PATH
        load_model = CONFIG.WEIGHT_PATH
        load_init_ckpt(load_model, sess)


        # extract feature
        sess.run(extract_iter.initializer)
        count = 0
        while True:
            try:
                vstart_time = time.time()
                batch_im, batch_label, batch_dpath = sess.run(extract_data)
                so = sess.run(score, feed_dict={x: batch_im,
                                                is_training: False, keep_prob: 1.0})
                for s, dpath in zip(so, batch_dpath):
                    np.save(dpath, s)
                vout_str = "Number: {}, time: {:.4f}"
                count += len(batch_label)
                print(vout_str.format(count, time.time() - vstart_time))
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    for phase in ['Train', 'Test']:
        extract_feat(phase)



