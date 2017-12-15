#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: lambdarank.py
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

with tf.name_scope("mlp"):
    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32, [None, config.FEATURE_NUM], name="X")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")
    if config.USE_HIDDEN_LAYER == True:
        with tf.name_scope("hidden_layer"):
            layer_h1 = tf.add(tf.matmul(X, weights["hidden"]), biases["hidden"])
            layer_h1 = tf.nn.relu(layer_h1)
        with tf.name_scope("out_layer"):
            y = tf.add(tf.matmul(layer_h1, weights["out"]), biases["out"])
    else:
        with tf.name_scope("linear_layer"):
            y = tf.add(tf.matmul(X, weights["linear"]), biases["linear"])

with tf.name_scope("matrices"):
    # score diff matrix with shape [doc_count, doc_count]
    # sigma_ij = matrix of sigma(s_i - s_j)
    #     in default RankNet, sigma = Identity, s_i = f(xi)
    # note: sigma_ij is the logit
    #    thus Pij = sigmoid(sigma_ij)
    sigma_ij = y - tf.transpose(y)

    # relevance diff matrix with shape [doc_count, doc_count]
    Sij_ = Y - tf.transpose(Y)
    # pairwise label matrix
    Sij = tf.minimum(1.0, tf.maximum(-1.0, Sij_))

    # pairwise label probability matrix
    Pij_hat = 1.0 / 2.0 * (1 + Sij)

    # pairwise lambda matrix
    # lambda_ij = dCij/ds_i = 1/2 * (1 - Sij) * dsigma(s_i - s_j)/d(s_i - s_j) * 1
    #    - (dsigma(s_i - s_j)/d(s_i - s_j) * 1) / (1 + e^(sigma_ij))
    # here we assign sigma = Identity, thus dsigma/d(si - sj) = 1
    lambda_ij = 1.0 / 2.0 * (1 - Sij) - 1.0 / (1 + tf.exp(sigma_ij))

    # the cost matrix
    # Cij = −P ̄ijoij + log(1 + eoij) = (1 − P ̄ij)oij + log(1 + e−oij)
    # can be rearrange as
    # Cij ≡ C(oij) = −P ̄ijlogPij − (1 − P ̄ij)log(1 − Pij)
    Cij = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=sigma_ij, labels=Pij_hat)

with tf.name_scope("train_op"):
    if config.QUALITY_MEASURE == config.NO_LAMBDA_MEASURE_USING_SGD:
        loss = tf.reduce_mean(Cij)
        train_op = tf.train.GradientDescentOptimizer(
            config.LEARNING_RATE).minimize(loss)
    elif config.QUALITY_MEASURE == config.LAMBDA_MEASURE_AUC:
        # factorize lambda_ij as lambda_i = \sum_{if Sij = 1}{lambda_ij} -
        # \sum_{if Sji = 1}{lambda_ji}
        pass
    elif config.QUALITY_MEASURE == config.LAMBDA_MEASURE_NDCG:
        # TODO
        pass

