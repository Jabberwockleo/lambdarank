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

USE_HIDDEN_LAYER = True
USE_TOY_DATA = True
LAYER_WIDTH = 10
FEATURE_NUM = 2
LEARNING_RATE = 0.01
TRAIN_BATCH_SIZE = 30
MODEL_PATH = "./model_ranknet.ckpt"
TRAIN_DATA = "./labeled.train"
TEST_DATA = "./labeled.test"
PREDICT_RESULT = "./labeled.predict"
if USE_TOY_DATA == True:
    TRAIN_DATA = "./toy.train"
    TEST_DATA = "./toy.test"
    PREDICT_RESULT = "./toy.predict"
MOCK_QUERY_DOC_COUNT = 4
