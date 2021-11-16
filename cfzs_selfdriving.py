import cv2
import numpy as np
import datetime
from lane_detector import lane_detector
from move_ctl import move_ctl
from camera import camera
from video_recorder import video_recorder
from yolov5_detector import yolov5_detector
from state_machine import state_machine
import time

if __name__ == "__main__":
    cam = camera(width=256, height=256, gap = 2)
    # move_ctl = move_ctl()
    state_machine = state_machine()
    lane_detector = lane_detector()
    yolo_detector = yolov5_detector()
    # rode_recorder = video_recorder(path = 'result.avi', width=256, height=256)  #记录行驶视频
    
    current_time = datetime.datetime.now()
    last_time = datetime.datetime.now()
    count = 0
    # for i in range(4):
    #     img = cam.getImg()
    #     result_boxes, result_scores, result_classid = yolo_detector.detect(img)
    #     time.sleep(1)
    #     image,distance = lane_detector.detect(img)
    #     time.sleep(1)

    while True:
        img = cam.getImg()
        # rode_recorder.write(img)
        # cv2.imshow('source_image',img)
        result_boxes, result_scores, result_classid = yolo_detector.detect(img)
        print(result_classid)
        # print(result_classid)
        image,distance = lane_detector.detect(img)
        # move_ctl.pid_ctl(position = distance, speed = 0)
        state_machine.run(img, distance, result_boxes, result_scores, result_classid)

        
        ##计算时间
        current_time = datetime.datetime.now()
        spend_time = (current_time-last_time).microseconds/1000000
        print("处理单张图片耗时：",spend_time,"  FPS:",1/spend_time)
        # print("distance：",distance)
        last_time = current_time
        #输出状态，保存行驶过程图片，方便debug
        yolo_detector.draw_boxes(img, result_boxes, result_scores, result_classid)
        cv2.imwrite('record_data/'+str(count)+'.jpg',image)
        count += 1
        # rode_recorder.write(img)
        # cv2.imshow('unet_image',image)
        cv2.waitKey(1)
        # cv2.imwrite('record_data/'+str(count)+'.jpg',image)
        # print(count,'  ',result_classid,'  ',result_boxes)
    
