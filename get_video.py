import cv2
import numpy as np
import datetime
from lane_detector import lane_detector
from move_ctl import move_ctl
from camera import camera
from video_recorder import video_recorder
from yolov5_detector import yolov5_detector

categories = ["go_straight", "go_and_right", "turn_left", "left_go_right", "zebra", "people", "obstacle", "traffic_light"]

if __name__ == "__main__":
    width = 1280
    height = 720
    cam = camera(width=width, height=height, gap = 2)
    yolo_recorder = video_recorder(path = 'yolo_detect_out.avi', width=width, height=height)  #记录行驶视频
    # unet_recorder = video_recorder(path = 'unet_detect_out.avi', width=256, height=256)  #记录行驶视频
    move_ctl = move_ctl()
    lane_detector = lane_detector()
    yolo_detector = yolov5_detector()
    
    current_time = datetime.datetime.now()
    last_time = datetime.datetime.now()
    count = 0

    while True:
        img = cam.getImg()
        # cv2.imshow('source_image',img)
        yolo_recorder.write(img)
        img = cv2.resize(img,(256,256))
        result_boxes, result_scores, result_classid = yolo_detector.detect(img)
        # print(result_classid)
        image,distance = lane_detector.detect(img)
        yolo_detector.draw_boxes(img, result_boxes, result_scores, result_classid)
        
        # unet_recorder.write(image)
        move_ctl.pid_ctl(position = distance, speed = 0.6)

        current_time = datetime.datetime.now()
        spend_time = (current_time-last_time).microseconds/1000000
        print("处理单张图片耗时：",spend_time,"  FPS:",1/spend_time)
        print("distance：",distance)
        last_time = current_time
        #输出状态，保存行驶过程图片，方便debug
        count += 1
        # cv2.imshow('unet_image',image)
        cv2.waitKey(1)
        # cv2.imwrite('record_data/'+str(count)+'.jpg',image)
        # print(count,'  ',result_classid,'  ',result_boxes)
    
