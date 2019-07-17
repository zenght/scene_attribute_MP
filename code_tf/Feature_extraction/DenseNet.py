"""Builds the DenseNet network."""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

import tensorflow as tf


def _variable_on_cpu(name, shape, para):
    """Helper to create a Variable stored on CPU memory.
    
    Args:
        name: name of the variable
        shape: list of ints
        para: parameter for initializer
    
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        if name == 'weights':
            # initializer = tf.truncated_normal_initializer(stddev=para, dtype=dtype)
            initializer = tf.contrib.layers.xavier_initializer(seed=1)
        else:
            initializer = tf.constant_initializer(para)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


class DenseNet(object):
    # Build the AlexNet model
    IMAGE_SIZE = 224    # input images size

    def __init__(self, model_flags):

        # Parse input arguments into class variables
        self.wd = 0.0001
        self.net_depth = model_flags.net_depth
        self.nlayers = model_flags.nlayers
        self.growth_rate = model_flags.growth_rate
        self.bn_size = model_flags.bn_size
        self.compressed = model_flags.compressed
        self.num_classes = model_flags.num_classes
        self.batch_size = model_flags.batch_size
        self.WEIGHTS_PATH = model_flags.weights_path
        self.num_epochs_per_decay = model_flags.num_epochs_per_decay
        self.learning_rate_decay_factor = model_flags.learning_rate_decay_factor
        self.lr = tf.Variable(model_flags.initial_learning_rate, trainable=False)
        self.lr_decay_op = self.lr.assign(
            self.lr * self.learning_rate_decay_factor)

    def inference(self, X, is_training, keep_prob):

        # First Convluation (224x224)
        x = conv(X, 7, 2*self.growth_rate, 2, pad=3, name='conv0')
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
        x = avgpool(x, 7, 1, padding='VALID')
        print(x.shape)
        x = tf.squeeze(x, [1, 2])
        print(x.shape)

        # classifier
        x = fc(x, self.num_classes, name='fc5')

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
                var = tf.get_variable(op_name)
                assign_ops.append(var.assign(data))
                # print("Loaded layer: {}".format(op_name))
                # try:
                #     # Biases
                #     var = tf.get_variable(op_name)
                #     assign_ops.append(var.assign(data))
                #     # print("Loaded layer: {}".format(op_name))
                # except:
                #     print('Not Load Layer: {}'.format(op_name))
                #     continue
                out_layers.append(layer)
        session.run(assign_ops)
        prt_strs = ["{} : {}".format(k, out_layers.count(k)) for k in sorted(set(out_layers))]
        for pstr in prt_strs:
            print(pstr)

        print('Loading the weights is Done in {:.4f}s.'.format(time.time() - st))

    def init_from_ckpt(self, weight_path=None):
        # This function is called before tf.global_variables_initializer()
        not_load_layers = []

        if self.WEIGHTS_PATH == 'None' and not weight_path:
            raise ValueError('Please supply the path to a checkpoint of model')
        elif weight_path:
            wpath = weight_path
            print('Loading the weights of {}'.format(wpath))
        else:
            wpath = self.WEIGHTS_PATH
            print('Loading the weights of {}'.format(self.WEIGHTS_PATH))

        cp_vars = tf.train.list_variables(wpath)
        load_layers = {}
        for var_name, _ in cp_vars:
            tmp_layer = var_name.split('/')[0]
            if tmp_layer not in not_load_layers:
                try:
                    tf.get_variable_scope().reuse_variables()
                    load_layers[var_name] = tf.get_variable(var_name)
                except:
                    continue

        print('----------Alreadly loaded variables--------')
        for k in load_layers:
            print(k)

        tf.train.init_from_checkpoint(wpath, load_layers)
        print('Loading the weights is Done.')

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
        
        Add summary for "Loss" and "Loss/avg".
        Args:
            logits: Logits is output of AlexNet model.
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]
        Returns:
            Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # the weight decay terms (L2 loss)
        # List of trainable variables of the layers we want to train
        train_var_list = [v for v in tf.trainable_variables()]
        print('********Weight Loss********')
        prt_list = []
        for train_var in train_var_list:
            if 'weights' in train_var.name:
                if self.wd is not None:
                    weight_decay = tf.multiply(tf.nn.l2_loss(train_var), self.wd, name='weight_loss')
                    tf.add_to_collection('losses', weight_decay)
                    # print(train_var.op.name)
                    prt_list.append(train_var.op.name.split('/')[0])
        # prt_strs = ["{} : {}".format(k, prt_list.count(k)) for k in sorted(set(prt_list))]
        # for pstr in prt_strs:
        #     print(pstr)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in AlexNet model.
        
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        
        Args:
            total_loss: Total loss from loss().
        Returns:
            loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        
        # Attach a scalar summary to all individual losses and the total loss
        # do the same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version 
            # of the loss as the original loss name.
            tf.summary.scalar(l.op.name + '(raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))
    
        return loss_averages_op

    def train(self, total_loss, num_examples, preload_layers, skip_layers, global_step):
        """Train AlexNet model.
        
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
        
        Args:
            total_loss: Total loss from loss().
            num_examples: numbers of examples per epoch for train
            global_step: Integer Variable counting the number of training steps
                         processed.
        Returns:
            train_op: op for training.
        """
        # Variables that affect learning rate.
        num_batches_per_epoch = num_examples / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)
        
        # Decay the learning rate exponentially based on the number of steps.
        self.lr = tf.train.exponential_decay(self.lr,
                                             global_step,
                                             decay_steps,
                                             self.learning_rate_decay_factor,
                                             staircase=True)
        tf.summary.scalar('learning_rate', self.lr)
        
        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(total_loss)

        # List of trainable variables of the layers we want to train
        train_var_list = [v for v in tf.trainable_variables()
                          if v.op.name.split('/')[0] in preload_layers
                          and v.op.name.split('/')[0] not in skip_layers]
        # print(train_var_list)
        # train_var_list = []
        train_var_list_random = [v for v in tf.trainable_variables()
                                 if v.op.name.split('/')[0] not in preload_layers
                                 and v.op.name.split('/')[0] not in skip_layers]
        # train_var_list_random = [v for v in tf.trainable_variables()]

        # Compute gradients of all trainable variable
        with tf.control_dependencies([loss_averages_op]):
            # opt = tf.train.GradientDescentOptimizer(self.lr)
            # opt_random = tf.train.GradientDescentOptimizer(self.lr * 10)
            # opt = tf.train.AdamOptimizer(self.lr)
            # opt = tf.train.AdadeltaOptimizer(self.lr)
            opt = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True)
            opt_random = tf.train.MomentumOptimizer(self.lr*10, 0.9, use_nesterov=True)
            grads = opt.compute_gradients(total_loss, train_var_list) if len(train_var_list) else []
            grads_random = opt_random.compute_gradients(total_loss, train_var_list_random)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads) if len(train_var_list) else []
        apply_random_gradient_op = opt_random.apply_gradients(grads_random, global_step=global_step)
        ops = [apply_gradient_op, apply_random_gradient_op] if len(train_var_list) else [apply_random_gradient_op]
        # Add histograms for trainable variables.
        print('Actual Train Variables')
        # for var in train_var_list + train_var_list_random:
        #     tf.summary.histogram(var.op.name, var)
        #     print("{} : {}".format(var.op.name, var.shape))

        # # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         tf.summary.histogram(var.op.name + '/gradients', grad)
        
        # Track the moving averages of all trainable variable.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #         self.moving_average_decay, global_step)
        # variables_averages_op = variable_averages.apply(train_var_list)
        # ops.append(variables_averages_op)
    
        # OPs for updating moving_mean and moving_variance in batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates_op = tf.group(*update_ops)
            ops.append(updates_op)

        with tf.control_dependencies(ops):
            train_op = tf.no_op(name='train')
    
        return train_op


"""
Predefine all necessary layer for CNN
"""


def DenseBlock(x, nlayers, bn_size, growth_rate, is_training, keep_prob, name):

    with tf.variable_scope(name):
        for i in range(nlayers):
            x = Denselayer(x, bn_size, growth_rate, is_training, keep_prob, name='denselayer{}'.format(i+1))

    return x


def Denselayer(x, bn_size, growth_rate, is_training, keep_prob, name):
    with tf.variable_scope(name):
        # bottleneck layer
        nx = BatchNorm(x, is_training, 'norm_1')
        nx = Relu(nx)
        nx = conv(nx, 1, bn_size*growth_rate, 1, 'conv_1')
        # print(nx.name)
        # norm layer
        nx = BatchNorm(nx, is_training, 'norm_2')
        nx = Relu(nx)
        nx = conv(nx, 3, growth_rate, 1, 'conv_2', pad=1)
        nx = dropout(nx, keep_prob)
        nx = tf.concat([x, nx], axis=3)

    return nx


def Transition(x, compressed, is_training, keep_prob, name):
    in_kernels = x.shape.as_list()[-1]
    # print(in_kernels*compressed)
    out_kernels = np.floor(in_kernels*compressed).astype(np.int32)
    with tf.variable_scope(name):
        x = BatchNorm(x, is_training, 'norm')
        x = Relu(x)
        x = conv(x, 1, out_kernels, 1, 'conv')
        x = dropout(x, keep_prob)
        x = avgpool(x, 2, 2)

    return x


def conv(x, kernel_size, num_kernels, stride_size, name, pad=0,
         with_bias=False, reuse=False, padding='VALID'):

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
    # print(x.get_shape())

    x = tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_size, stride_size, 1],
                                         padding=padding)

    with tf.variable_scope(name, reuse=reuse):
        # Create tf variables for the weights and biases of the conv layer
        weights = _variable_on_cpu('weights', [kernel_size, kernel_size,
                                               input_channels, num_kernels], 1e-1)

        # Apply convolution function
        conv = convolve(x, weights)

        if with_bias:
            # Add biases
            biases = _variable_on_cpu('biases', [num_kernels], 0.0)
            conv = tf.nn.bias_add(conv, biases)

        return conv


def BatchNorm(x, is_training, name):
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-5, scale=True,
                                        is_training=is_training, fused=True,
                                        zero_debias_moving_mean=False, scope=name)


def Relu(x):
    return tf.nn.relu(x)


def fc(x, num_out, name, reuse=False,
       relu=False, batch_norm=False, is_training=False):
    num_in = x.shape.as_list()[-1]
    with tf.variable_scope(name, reuse=reuse) as scope:

        # Create tf variable for the weights and biases
        weights = _variable_on_cpu('weights', [num_in, num_out], 1e-1)
        biases = _variable_on_cpu('biases', [num_out], 1.0)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if batch_norm:
            # Adds a Batch Normalization layer
            act = tf.contrib.layers.batch_norm(act, center=True, scale=True,
                                               trainable=True, is_training=is_training,
                                               reuse=reuse, scope=scope)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, kernel_size, stride_size, pad=0, padding='VALID'):
    x = tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding)


def avgpool(x, kernel_size, stride_size, padding='VALID'):
    return tf.nn.avg_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


