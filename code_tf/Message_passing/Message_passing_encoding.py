# -*- coding: utf-8 -*-
"""
Created on 09/05/2018

Description: Aggregate ImageNet-based and Places-based features of MIT67 by using Message Passing

Author: Haitao Zeng

"""

from __future__ import absolute_import
from __future__ import division
from sklearn.svm import LinearSVC
from datetime import datetime
import os
import sys
import time
import scipy.io as sio
import numpy as np
import math
import yaml
from easydict import EasyDict as edict
# tmp_dir = '/home/cgw/Data2-CGW/tmp/TF'
tmp_dir = '/home/haitaizeng/stanforf/song_works'


def readconfig(config_path):
    with open(config_path, 'r') as f:
        CONFIG = edict(yaml.load(f))
    return CONFIG


class MIT_conf:
    # store all configurations for MIT67 dataset

    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    configpath = os.path.join(dir_path, "configs/MIT67_MP.yaml")

    CONFIG = readconfig(configpath)
    FeatureList = CONFIG.FEATURE.NAME
    imgSize = CONFIG.FEATURE.SIZE


    # some parameters for MRF
    lamb = CONFIG.PARAMETER.LAMB
    win_size = CONFIG.PARAMETER.WINSIZE  # Default is 6
    max_iter_MRF = CONFIG.PARAMETER.ITER_MRF # for MRF max iter number
    max_iter_MP = CONFIG.PARAMETER.ITER_MP  # for Message Passing max iter number
    min_inc = CONFIG.PARAMETER.MIN_INC
    learning_rate = CONFIG.PARAMETER.LR
    W_LIST = np.array(CONFIG.PARAMETER.W_LIST)


def read_list(file_path):
    d_list = []
    d_label = []
    with open(file_path, 'r') as file:
        for line in file:
           # MIT67
            d_list.append(line.split()[0])
            d_label.append(line.split()[1])
        d_list = np.array(d_list)
        d_label = np.array(d_label)
    return d_list, d_label

def SVM(train_feats, train_labels, test_feats, test_labels):
    begintime = time.time()
    clf = LinearSVC(C=1)

    print('*****Starting train Linear SVM model*****')
    clf.fit(train_feats, train_labels)
    print('\n*****Starting test model*****')
    acc = clf.score(test_feats, test_labels)
    acc = acc * 100
    print('\033[1;36;48m The accuracy : {:.2f} \033[0m'.format(acc))
    print('TIME COST = {:.2f}\n'.format(time.time() - begintime))


def smn_entropy(smn):
    # compute entropy of SMNs
    ent = -np.sum(smn * np.log2(smn + 1e-4))
    return ent


def geodesic_entrogy(cur_smn, aux_smn, lamb=MIT_conf.lamb):
    # cur_smn'size is [feature_dim], aux_smn'size is [feature_dim, patch_num]
    # lamb = MIT_conf.lamb  # lamb = 0.005
    temp_smn = np.sqrt(cur_smn).dot(np.sqrt(aux_smn))
    # temp_smn = sqrtt(cur_smn).dot(sqrtt(aux_smn))
    if (temp_smn >= 1).any():
        r = (temp_smn >= 1).nonzero()
        temp_smn[r] = 1 - 1e-6

    temp_smn_cos = np.arccos(temp_smn)
    ent = lamb * smn_entropy(cur_smn)
    gdist = np.sum(temp_smn_cos) + ent
    temp_smn_r = 1 / np.sqrt(1 - temp_smn ** 2 + 1e-6)
    gradent = lamb * (1 + np.log2(cur_smn + 1e-4))
    grad = gradent + np.mean(np.sqrt(aux_smn) * temp_smn_r, 1)
    grad = grad / np.sum(np.abs(grad))
    return gdist, grad


def euclid_dist(A, B):
    # A, B need to be 2-dim
    eps = 1e-12
    # print(A, B.T)
    M = np.dot(A, B.T)
    H = np.square(A).sum(axis=1, keepdims=True)
    K = np.square(B).sum(axis=1, keepdims=True)
    D = -2 * M + H + K.T
    return D, np.sqrt(D + eps)


def ms_find_neighbor(cind, l, msf_inds, msf_rowNum, win_size):
    # find the index of the Before and after the feature of the current feature
    tind = []
    wnum = (win_size * 2 + 1) ** 2
    for tl, msfi in enumerate(msf_inds):
        # print("tl , l",tl,l)
        if msfi.ndim == 1:
            msfi = msfi[None, :]
        if tl == l:
            # this is different from TIP_song in boundary point
            nb_dist, _ = euclid_dist(cind[None, :], msfi)
            nb_num = wnum
            nb_sort = nb_dist[0].argsort()
            nb_ind = nb_sort[1:nb_num]
        elif tl == l - 1:
            # this is different from TIP_song in parent neighbor number
            # In TIP_song, nb_num is win_size**2 (copies), here is 1, no copy
            proj_ind = cind * msf_rowNum[tl] / msf_rowNum[l]
            nb_dist, _ = euclid_dist(proj_ind[None, :], msfi)
            nb_num = 1
            nb_sort = nb_dist[0].argsort()
            nb_ind = np.tile(nb_sort[:nb_num], [wnum])
        elif tl == l + 1:
            proj_ind = cind * msf_rowNum[tl] / msf_rowNum[l]
            nb_dist, _ = euclid_dist(proj_ind[None, :], msfi)
            nb_num = wnum
            nb_sort = nb_dist[0].argsort()
            nb_ind = nb_sort[:nb_num]
        else:
            nb_ind = []
        tind.append(nb_ind)
    return tind


def ms_neighbor_ind(CONFIG, win_size=MIT_conf.win_size):
    msf_rowNum = []
    msf_inds = []

    for lsize in MIT_conf.imgSize:
        feat_location = os.path.join(CONFIG.FEATURE.DIR, CONFIG.FEATURE.NAME[0])
        feat_location = os.path.join(feat_location, lsize)
        feat_location = os.path.join(feat_location, os.listdir(feat_location)[0])


        feat = np.load(feat_location)        # print(feat.shape)
        if feat.ndim == 1:
            feat = feat[:, None]
        elif feat.ndim == 3:
            feat = feat.reshape(64, feat.shape[2])
            feat = feat.transpose()
            # feat = np.mean(feat, axis=1, keepdims=True)
        # print(feat.shape)
        pnum = feat.shape[1]
        cur_rnum = np.sqrt(pnum).astype(np.int8)  # Rounding off PNUM
        msf_rowNum.append(cur_rnum)
        # what's the meaning of this line
        cur_c = np.arange(pnum) // cur_rnum
        cur_r = np.arange(pnum) - cur_c * cur_rnum
        cur_inds = np.concatenate([cur_r[:, None], cur_c[:, None]], axis=1) + 0.5
        msf_inds.append(cur_inds)

        # print('{} : {}'.format(l, msf_inds))
    ms_nb_ind = []

    for l, cur_inds in enumerate(msf_inds):
        nb_ind = []
        for cind in cur_inds:
            # print("CIndex",cind)
            tind = ms_find_neighbor(cind, l, msf_inds, msf_rowNum, win_size)
            nb_ind.append(tind)
        # print(nb_ind)
        ms_nb_ind.append(nb_ind)
    # ms_nb_ind has three dim, first for target scale, second for patch in scale,
    # third for neighbor scale
    return ms_nb_ind


def neighbor_smn(msf_smn, amsf_smn, nb_ind):
    tsmn = []
    for l, nind in enumerate(nb_ind):
        if len(nind) == 0:
            continue
        else:

            smn1 = msf_smn[l][:, :, nind]
            k = smn1.shape[0]
            # print(smn1.shape)
            smn1 = np.concatenate(np.split(smn1, k), axis=2)[0]
            smn2 = amsf_smn[l][:, nind]
            tsmn.append(smn1)
            tsmn.append(smn2)
    tsmn = np.concatenate(tsmn, axis=1)
    # print(tsmn.shape)
    return tsmn


def MP_main(Dtype, t_path, CONFIG):
    # improve representation of patch SMN by using surround patches
    Feature_path = CONFIG.FEATURE.DIR
    list, test_label = read_list(t_path)
    Length = range(len(list))
    print len(list)
    all_smn = np.ndarray([CONFIG.DATASET.TOTAL, CONFIG.DATASET.N_CLASSES])
    all_up_smn = np.ndarray([CONFIG.DATASET.TOTAL, CONFIG.DATASET.N_CLASSES])
    print('Starting process patch SMNs')
    ms_nb_ind = ms_neighbor_ind(CONFIG)
    cnt = 1
    disp_step = 100
    start_time = time.time()
    for num, line in enumerate(list):
        if num%100==1:
            print("{} images have done".format(num))
        if Dtype =='Test':
            # num += 19850 #SUN397
            num += 5360 # MIT67


        msf_smn = []  # multi-scale, multi-feature smn
        amsf_smn = []  # aggregating multi-feature smn by using Fweights
        msf_rowNum = []

        for l, image_size in enumerate(MIT_conf.imgSize):
            mf_smn = []
            for pretrained in MIT_conf.FeatureList:
                source_path = os.path.join(Feature_path, '{:}/{:}'.
                                           format(pretrained, image_size))
                fnewname = '_'.join(line[:-4].split('/'))

                featurepath = os.path.join(source_path, fnewname + '.npy')
                feature = np.load(featurepath)
                if feature.ndim == 1:
                    feature = feature[:, None]
                elif feature.ndim == 3:
                    feature = feature.reshape(64, feature.shape[2])
                    feature = feature.transpose()
                mf_smn.append(feature)
            mf_smn = np.array(mf_smn)
            amf_smn = np.sum(mf_smn *MIT_conf.W_LIST[:, l][:, None, None], axis=0) / np.sum(MIT_conf.W_LIST[l])
            pnum = amf_smn.shape[1]
            msf_rowNum.append(np.sqrt(pnum).astype(np.int8))
            msf_smn.append(mf_smn)
            amsf_smn.append(amf_smn)
        new_amsf_smn = amsf_smn[:]

        for ii in range(MIT_conf.max_iter_MP):
            # len(MIT_conf.imgSize) = 0,1
            for l in range(len(MIT_conf.imgSize)):
                # print(amsf_smn[l].shape)
                pnum = amsf_smn[l].shape[1]
                # pnum=64,or =1, 就是64个特征点,找他们对应的邻居节点的坐标
                for j in range(pnum):
                    cur_smn = amsf_smn[l][:, j]
                    # nb_ind is a list of mult-scale neighbor index, a list of len(scales)
                    #  get the index of the neighbor?
                    nb_ind = ms_nb_ind[l][j]
                    # nb_smns is all neighbor smns of current smn in multi-scale
                    nb_smns = neighbor_smn(msf_smn, amsf_smn, nb_ind)
                    # Caculate the GD of all neighbor smns and current smn
                    gdist, grad = geodesic_entrogy(cur_smn, nb_smns, lamb = MIT_conf.lamb)
                    cur_ent = gdist.sum()
                    # print(cur_ent)
                    pre_ent = 1e10
                    iii = 0
                    while iii < MIT_conf.max_iter_MRF and cur_ent < (pre_ent - MIT_conf.min_inc):
                        cur_smn = np.sqrt(cur_smn) + grad * MIT_conf.learning_rate
                        cur_smn = np.sign(cur_smn) * np.square(cur_smn)
                        if cur_smn.min() < 0:
                            cur_smn = cur_smn - cur_smn.min()
                        cur_smn = cur_smn / cur_smn.sum()
                        gdist, grad = geodesic_entrogy(cur_smn, nb_smns, lamb = MIT_conf.lamb)
                        pre_ent = cur_ent
                        cur_ent = gdist.sum()
                        iii += 1
                    new_amsf_smn[l][:, j] = cur_smn

            amsf_smn = new_amsf_smn[:]
        # total smn
        total_smn = new_amsf_smn[-1].mean(axis=1)
        up_smn = new_amsf_smn[0].mean(axis=-1)
        all_smn[num, :] = total_smn
        all_up_smn[num, :] = up_smn

        cnt += 1
    if Dtype == 'Train':
        return all_up_smn[Length, :],test_label
    elif Dtype=='Test':
        # test_feat = all_smn[Length, :]
        # test_feat = np.exp(test_feat) / sum(np.exp(test_feat))
        # pred_label = test_feat.argmax(axis=1)

        test_feat224 = all_up_smn[range(5360,6700),:]
        test_feat224 = np.exp(test_feat224) / sum(np.exp(test_feat224))
        pred_label224 = test_feat224.argmax(axis=1)
        print pred_label224
        # if the database is MIT67 Then use np.uint8
        ACC_224 = (np.uint8(test_label) == pred_label224).mean()
        print('Time Cost: {:.2f}s'.format((time.time() - start_time)))
        print('\033[1;34;48m Val_acc: {:.4f}\033[0m'.format(ACC_224))

        return test_feat224, test_label, ACC_224




def attribute_classification(SVM=False):
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    configpath = os.path.join(dir_path, "configs/MIT67_MP.yaml")
    CONFIG = readconfig(configpath)
    BEGIN_TIME = time.time()
    print("This epoch begin at:{}".format(datetime.now()))
    # mit67_path
    train_file = CONFIG.DATASET.SPLIT.TRAIN
    test_file = CONFIG.DATASET.SPLIT.TEST

    if SVM:
        test_feats, test_labels, Accuracy = MP_main('Test', test_file, CONFIG)
        train_feats, train_labels = MP_main('Train', train_file, CONFIG)
        SVM(train_feats,train_labels,test_feats,test_labels)
    else:
        _, _, Accuracy = MP_main('Test', test_file, CONFIG)
        print("Accuracy", Accuracy)
    END_TIME = time.time()
    print ("Time cost: {}".format(END_TIME-BEGIN_TIME))


if __name__ == '__main__':
    attribute_classification()



