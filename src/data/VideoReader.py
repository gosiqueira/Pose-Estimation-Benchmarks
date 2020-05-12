#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2


class VideoReader(object):
    def __init__(self, video_filepath=None):
        if video_filepath:
            self.reader = cv2.VideoCapture(video_filepath)
        else:
            self.reader = None
        
    def get_frame(self):
        if self.reader:
            if self.reader.isOpened():
                ret, frame = self.reader.read()
                return ret, frame
            else:
                raise StopIteration
        else:
            raise RuntimeError

    def release(self):
        if self.reader:
            self.reader.release()
        else:
            raise RuntimeError

    def set_video(self, video_filepath):
        self.reader = cv2.VideoCapture(video_filepath)

    def is_opened(self):
        if self.reader:
            return self.reader.isOpened()
        else:
            raise RuntimeError
