#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2
import logging
import os
import time

from methods import model_zoo
from utils.AverageMeter import AverageMeter

LOG_FORMAT = '[%(levelname)s - %(name)s]: %(asctime)s - %(message)s'
POSE_ESTIMATORS = ['AlphaPoseMXNet', 'DetectronCocoKeypoints', 'SimpleBaselineMXNet']

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--method', help='Pose estimator method name', type=str, choices=['AlphaPoseMXNet', 'DetectronCocoKeypoints', 'SimpleBaselineMXNet', 'all'], default='all')
    parser.add_argument('-f', '--folder', dest='video_folder', help='Path to benchmark videos folder', type=str, required=True)

    return parser.parse_args()
    

def gen_benchmark(pose_estimator, video_folder):
    logging.basicConfig(filename='pose_estimation_benchmarks.log', filemode='w', format=LOG_FORMAT, level=logging.INFO)
    logging.info(f'Evaluating {pose_estimator} method...')

    tic = time.time()

    net = model_zoo.get_model(pose_estimator)
    fps_meter = AverageMeter()

    videos_list = os.listdir(video_folder)

    for video in videos_list:
        logging.info(f'Running video {video}...')
        video_filepath = os.path.join(video_folder, video)
        video_reader = cv2.VideoCapture(video_filepath)
        while(video_reader.isOpened()):
            ret, frame = video_reader.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                model_in = time.time()
                net.get_poses(frame)
                fps_meter.update(1.0 / (time.time() - model_in))
            else:
                break

        logging.info(f'Average FPS: {fps_meter.avg}')

    tac = time.time()
    logging.info(f'Done in {(tac - tic):.3f} seconds.')


if __name__ == '__main__':
    args = get_args()

    if args.method == 'all':
        for pose_estimator in POSE_ESTIMATORS:
            gen_benchmark(pose_estimator, args.video_folder)

    else:
        gen_benchmark(POSE_ESTIMATORS[args.method], args.video_folder)
