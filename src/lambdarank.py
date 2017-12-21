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

    def graph_params():
        """Get layer params in array

        Returns:
            layer_params: list of params of params in all layers.
                One layer may contain multiple lists, such as
                weight_1 = [w_1, w_2, .., w_n] bias_1 = [w_n+1, .., w_p]
                weight_2 = [w_p+1, ...]
        """
        layer_params = []
        if config.USE_HIDDEN_LAYER == True:
            layer_params.append((weights["hidden"], biases["hidden"]))
            layer_params.append((weights["out"], biases["out"]))
        else:
            layer_params.append((weights["linear"], biases["linear"]))
        return layer_params


    def compute_graph(X):
        """Build compute graph

        define a function for computing ds_i/dw_k respectively,
            as the tf.gradient() computes sum_{k} dy_k/dx_i w.r.t x_i

        Args:
            X: the input feature vector tensor shaped [None, x_i]
        Returns:
            y: the output predict tensor shaped [None, y_i]
        """
        if config.USE_HIDDEN_LAYER == True:
            with tf.name_scope("hidden_layer"):
                layer_h1 = tf.add(tf.matmul(X, weights["hidden"]), biases["hidden"])
                layer_h1 = tf.nn.relu(layer_h1)
            with tf.name_scope("out_layer"):
                y = tf.add(tf.matmul(layer_h1, weights["out"]), biases["out"])
        else:
            with tf.name_scope("linear_layer"):
                y = tf.add(tf.matmul(X, weights["linear"]), biases["linear"])
        return y

with tf.name_scope("matrices"):
    y = compute_graph(X)
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

    # dC/dw_k = \sum{lambda_ij * (ds_i/dw_k - ds_j/dw_k)}
    #    = \sum{lambda_i * ds_i/dw_k}
    # lambda_i is the coefficiency of dC/dw_k
    # which is factorized as lambda_i = \sum_{if Sij = 1}{lambda_ij} -
    #    \sum_{if Sji = 1}{lambda_ji}
    ij_positive_label_mat = tf.maximum(Sij, 0) #Mij = 1 if (i, j) \in P
    ij_sum_mat = ij_positive_label_mat * lambda_ij
    ij_sum = tf.reduce_sum(ij_sum_mat, [1])
    ji_sum = tf.reduce_sum(ij_sum_mat, [0])
    lambda_i = ij_sum - ji_sum #lambda_i for \sum_{i}dCij/dsi - \sum_{i}dCji/dsj

with tf.name_scope("train_op"):
    t = tf.constant(0) # debug variable
    loss = tf.reduce_mean(Cij)
    if config.QUALITY_MEASURE == config.NO_LAMBDA_MEASURE_USING_SGD:
        train_op = tf.train.GradientDescentOptimizer(
            config.LEARNING_RATE).minimize(loss)
    elif config.QUALITY_MEASURE == config.LAMBDA_MEASURE_AUC:
        # unpack X on dimension 0 and computes gradients w.r.t x_i,
        # resulting on a tensor R with R.shape[0] = X.shape[0]
        # because tf.gradients(Y, X) computes sum_{k} dy_k/dx_i
        # we want ds_i/dw_k, i.e. dg(x_i)/dw_k
        # thus we need to compute s_i and w_k respectively
        def make_dsi_dwk_closure(w_k):
            """Make a closure w.r.t wk

            Args:
                w_k: respected to which gradient is computed
            Returns:
                a function passed to tf.map_fn() which accept x_i
            """
            def compute_dsi_dwk(x_i):
                """Compute gradient of graph(x_i) w.r.t w_k

                Args:
                    x_i: single input feature vector
                Returns:
                    a single tensor representing gradient ds_i/dw_k
                """
                xi_mat = tf.expand_dims(x_i, 0)
                return tf.gradients(compute_graph(xi_mat), [w_k])[0]
            return compute_dsi_dwk


        # flatten params of layer params
        layer_params = graph_params()
        wk_arr = [wk for w_b in layer_params for wk in w_b]

        # computes [None, gradient] matrix of ds_i/dw_k
        def compute_ds_dwk(w_k):
            """Compute [ds_1/dw_k, ds_2/dw_k, ..] mat

            Args:
                w_k: a node param to compute
            Returns:
                a [None, gradient] tensor
            """
            dsi_dwk_mat = tf.map_fn(make_dsi_dwk_closure(w_k), X)
            return dsi_dwk_mat

        # compute gradients dC/dw_k
        dC_dwk_arr = []
        for wk in wk_arr:
            # ds_dwk for hidden layer param shaped [N, feature_num, layer_width]
            # ds_dwk for output layer param shaped [N, feature_num]
            ds_dwk = compute_ds_dwk(wk)
            # dC/dwk is \sum_{i} lambda_i * ds_i/dwk
            lambdai_dsi_dwk = tf.map_fn(lambda x: x[0] * x[1],
                    (ds_dwk, tf.expand_dims(lambda_i, 1)), dtype=tf.float32)
            dC_dwk = tf.reduce_sum(lambdai_dsi_dwk, axis=[0])
            dC_dwk_arr.append(dC_dwk)

        # flattened
        flat_wk = wk_arr[:]
        flat_grad = dC_dwk_arr[:]

        # apply gradients
        train_op = tf.train.GradientDescentOptimizer(
            config.LEARNING_RATE).apply_gradients(
                [(gr, wk) for gr, wk in zip(flat_grad, flat_wk)]
            )
        pass
    elif config.QUALITY_MEASURE == config.LAMBDA_MEASURE_NDCG:
        # TODO
        pass

