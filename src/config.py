#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: config.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""

LAMBDA_MEASURE_MRR = "mean_reciprocal_rank"
LAMBDA_MEASURE_MAP = "mean_average_precision"
LAMBDA_MEASURE_MAP = "expected_reciprocal_rank"
LAMBDA_MEASURE_NDCG = "normalized_discounted_cumulative_gain"
LAMBDA_MEASURE_AUC = "factorized_pairwise_precision"
NO_LAMBDA_MEASURE_USING_SGD = "pure_sgd"

DEBUG_LOG = True
QUALITY_MEASURE = NO_LAMBDA_MEASURE_USING_SGD
QUALITY_MEASURE = LAMBDA_MEASURE_AUC
QUALITY_MEASURE = LAMBDA_MEASURE_NDCG
USE_HIDDEN_LAYER = True
USE_TOY_DATA = True
LAYER_WIDTH = 10
FEATURE_NUM = 2
LEARNING_RATE = 0.001
MODEL_PATH = "./model_lambdarank.ckpt"
TRAIN_DATA = "./labeled.train"
TEST_DATA = "./labeled.test"
PREDICT_RESULT = "./labeled.predict"
if USE_TOY_DATA == True:
    TRAIN_DATA = "./toy.train"
    TEST_DATA = "./toy.test"
    PREDICT_RESULT = "./toy.predict"
MOCK_QUERY_DOC_COUNT = 4
