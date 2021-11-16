#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import cv2

class video_recorder(object):
    def __init__(self, path = 'road.avi', width=256, height=256):
        # 指定视频编解码方式为MJPG
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        fps = 20.0 # 指定写入帧率为20
        frameSize = (width,height) # 指定窗口大小
        # 创建 VideoWriter对象
        self.out = cv2.VideoWriter(path, codec, fps, frameSize)
        self.flip = False

    def write(self, img):
        if self.flip:
            img = cv2.flip(img, -1)
        self.out.write(img)