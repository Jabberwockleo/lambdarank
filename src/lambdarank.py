#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: ranknet.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""

import tensorflow as tf
import config

weights = {
        "hidden": tf.Variable(tf.random_normal(
            [config.FEATURE_NUM, config.LAYER_WIDTH]), name="W_hidden"),
        "out": tf.Variable(tf.random_normal([config.LAYER_WIDTH, 1]), name="W_out"),
        "linear": tf.Variable(tf.random_normal([config.FEATURE_NUM, 1]), name="W_linear")
}

biases = {
        "hidden": tf.Variable(tf.random_normal([config.LAYER_WIDTH]), name="b_hidden"),
        "out": tf.Variable(tf.random_normal([1]), name="b_out"),
        "linear": tf.Variable(tf.random_normal([1]), name="b_linear"),
}

with tf.name_scope("mlp1"):
    with tf.name_scope("input"):
        X1 = tf.placeholder(tf.float32, [None, config.FEATURE_NUM], name="X1")
        O1 = tf.placeholder(tf.float32, [None, 1], name="O1")
    if config.USE_HIDDEN_LAYER == True:
        with tf.name_scope("hidden_layer"):
            layer_h1 = tf.add(tf.matmul(X1, weights["hidden"]), biases["hidden"])
            layer_h1 = tf.nn.relu(layer_h1)
        with tf.name_scope("out_layer"):
            o1 = tf.add(tf.matmul(layer_h1, weights["out"]), biases["out"])
    else:
        with tf.name_scope("linear_layer"):
            o1 = tf.add(tf.matmul(X1, weights["linear"]), biases["linear"])

with tf.name_scope("mlp2"):
    with tf.name_scope("input"):
        X2 = tf.placeholder(tf.float32, [None, config.FEATURE_NUM], name="X2")
        O2 = tf.placeholder(tf.float32, [None, 1], name="O2")
    if config.USE_HIDDEN_LAYER == True:
        with tf.name_scope("hidden_layer"):
            layer_h2 = tf.add(tf.matmul(X2, weights["hidden"]), biases["hidden"])
            layer_h2 = tf.nn.relu(layer_h2)
        with tf.name_scope("out_layer"):
            o2 = tf.add(tf.matmul(layer_h2, weights["out"]), biases["out"])
    else:
        with tf.name_scope("linear_layer"):
            o2 = tf.add(tf.matmul(X2, weights["linear"]), biases["linear"])

with tf.name_scope("loss"):
    O12 = O1 - O2
    o12 = o1 - o2

    #Numerical TRICK:
    #  log(1 + exp(-x)) = log(1 + exp(x)) - x
    #  i.e. 1 + exp(-x) = exp(log(1 + exp(-|x|)) - min(0, x))

    #pred = 1 / (1 + tf.exp(-o12)) #maybe numerical overflow
    pred = 1 / tf.exp(-tf.minimum(o12, 0) + tf.log(1 + tf.exp(-tf.abs(o12)))) #reformulation

    #truth = 1 / (1 + tf.exp(-O12)) #maybe numerical overflow
    truth = 1 / tf.exp(-tf.minimum(O12, 0) + tf.log(1 + tf.exp(-tf.abs(O12)))) #reformulation

    cross_entropy = -truth * tf.log(tf.clip_by_value(pred, 1e-8, 1.0))
    - (1 - truth) * tf.log(tf.clip_by_value(1 - pred, 1e-8, 1.0))
    reduce_sum = tf.reduce_sum(cross_entropy, 1)
    loss = tf.reduce_mean(reduce_sum)

with tf.name_scope("train_op"):
    train_op = tf.train.GradientDescentOptimizer(config.LEARNING_RATE).minimize(loss)

