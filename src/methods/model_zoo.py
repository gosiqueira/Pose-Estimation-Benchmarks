#!/usr/bin/env python
# -*- coding: utf-8 -*-

from methods.alphapose.mxnet.AlphaPoseMXNet import AlphaPoseMXNet
from methods.simple_baselines.mxnet.SimpleBaselinesMXNet import SimpleBaselinesMXNet
from methods.detectron_coco_keypoints.DetectronCocoKeypoints import DetectronCocoKeypoints

def get_model(model_str_name):
    if model_str_name == 'AlphaPoseMXNet':
        return AlphaPoseMXNet()
    elif model_str_name == 'DetectronCocoKeypoints':
        return DetectronCocoKeypoints()
    elif model_str_name == 'SimpleBaselinesMXNet':
        return SimpleBaselinesMXNet()
    else:
        raise NotImplementedError
