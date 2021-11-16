#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

class camera(object):
    def __init__(self, width=256, height=256, gap = 2):
        def gstreamer_pipeline(
                display_width=1280,
                display_height=720,
                capture_width=1280,
                capture_height=720,
                framerate=40,
                flip_method=0,
        ):
            return (
                    "nvarguscamerasrc ! "
                    "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
                    "nvvidconv flip-method=%d ! "
                    "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                    "videoconvert ! "
                    "video/x-raw, format=(string)BGR ! "
                    "appsink"
                    % (
                        capture_width,
                        capture_height,
                        framerate,
                        flip_method,
                        display_width,
                        display_height,
                    )
            )
        self.cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0,display_width=width,display_height=height), cv2.CAP_GSTREAMER)
        self.gap = gap
        if not self.cam.isOpened():
            print('cannot open camera')
            exit(1)
        success, img = self.cam.read()
        if not success:
            print('read camera error')
            exit(1)
    def getImg(self):
        for i in range(self.gap):
            success, img = self.cam.read()
        if not success:
            print('read camera error')
            exit(1)
        return img
