#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: train.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""

import tensorflow as tf
import numpy as np
import lambdarank
import mock
import config

if config.USE_TOY_DATA == True:
    fin = open(config.TRAIN_DATA, "w")
    mock.generate_labeled_data_file(fin, 10000)
    fin.close()

fout = open(config.TRAIN_DATA, "r")
train_data, train_data_keys = mock.parse_labeled_data_file(fout)
fout.close()

train_data_key_count = len(train_data_keys)

def convert_np_data(query_doc_list):
    """Convert query doc list to numpy data of one retrival

    Args:
        query_doc_list: list of list: [score, f1, f2 , ..., fn]

    Return:
        X, Y: [feature_vec], [label]
    """
    x = []
    y = []
    for qd in query_doc_list:
        x.append(qd[1:])
        y.append(qd[:1])

    return np.array(x), np.array(y)

saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    query_doc_index = 0
    for epoch in range(0, 10000):
        X, Y = [[], []], [[], []]
        # get next query_doc list by a query
        query_doc_index += 1
        query_doc_index %= len(train_data_keys)
        key = train_data_keys[query_doc_index]
        doc_list = train_data[key]
        # convert to graph input structure
        X, Y = convert_np_data(doc_list)
        sess.run(lambdarank.train_op, feed_dict={lambdarank.X:X, lambdarank.Y:Y})
        if epoch % 100 == 0:
            loss, \
                    debug_X, debug_Y, debug_y,\
                    debug_sigma_ij, debug_Sij = \
                    sess.run([lambdarank.loss,
                        lambdarank.X,
                        lambdarank.Y,
                        lambdarank.y,
                        lambdarank.sigma_ij,
                        lambdarank.Sij],
                       feed_dict={lambdarank.X:X, lambdarank.Y:Y})
            print "-- epoch[%d] loss[%f] -- " % (
                epoch,
                loss,
            )

        if epoch % 1000 == 0 and config.DEBUG_LOG == True:
            print "X:\n", debug_X
            print "Y:\n", debug_Y
            print "y:\n", debug_y
            print "sigma_ij:\n", debug_sigma_ij
            print "Sij:\n", debug_Sij
    save_path = saver.save(sess, config.MODEL_PATH)
    print("Model saved in file: %s" % save_path)
