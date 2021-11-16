import cv2
import numpy as np
import math
import tensorrt as trt
from lane_detector.unet_trt_inference import *

is_visualize = False
resize_width = 256
resize_height = 256
def hough_lines(img,mask, rho=1, theta=np.pi / 180, threshold=10, min_line_len=10, max_line_gap=10):
    #加掩码
    # masked_img = cv2.bitwise_and(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
    masked_img = cv2.bitwise_and(img, mask)
    #根据阈值生成二值图像
    ret,bin_img = cv2.threshold(masked_img,50,255,cv2.THRESH_BINARY)
    if is_visualize : cv2.imshow('binary_img',bin_img)

    # rho = 1       # 霍夫像素单位
    # theta = np.pi / 180    # 霍夫角度移动步长
    # hof_threshold = 20   # 霍夫平面累加阈值threshold
    # min_line_len = 30    # 线段最小长度
    # max_line_gap = 60    # 最大允许断裂长度
    # rho：线段以像素为单位的距离精度
    # theta : 像素以弧度为单位的角度精度(np.pi/180较为合适)
    # threshold : 霍夫平面累加的阈值
    # minLineLength : 线段最小长度(像素级)
    # maxLineGap : 最大允许断裂长度：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大

    lines = cv2.HoughLinesP(bin_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if is_visualize : 
        houph = np.zeros([resize_height,resize_width], dtype=np.uint8)
        for line in lines:
            lane = line[0]
            x1,y1,x2,y2 = lane[0],lane[1],lane[2],lane[3]
            cv2.line(houph,(x1,y1),(x2,y2),255,2)
        cv2.imshow('houph',houph)
    return lines
#计算车位置偏离中心的距离（计算理想中心到车道线中位线的距离）
def get_distance_from_point_to_line(point, p_position):
    line_point1 = ((p_position[0][0]+p_position[1][0])/2,p_position[0][1])
    line_point2 = ((p_position[2][0]+p_position[3][0])/2,p_position[2][1])
    #计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    #根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    middle_x = int((line_point1[0] + line_point2[0])/2)
    state = 'left'
    if middle_x <= int(256/2):
        state = 'left'
        distance = abs(distance)
    else:
        distance = 0-abs(distance)
        state = 'right'

    #可视化
    # if is_visualize:
    #     if B != 0:
    #         slope = A/B
    #         point = (int(point[0]/Wrate),int(point[1]/Hrate))
    #         line_point1 = (int(line_point1[0]/Wrate),int(line_point1[1]/Hrate))
    #         line_point2 = (int(line_point2[0]/Wrate),int(line_point2[1]/Hrate))
    #         temp = -1/slope
    #         vslope = math.atan(temp)
    #         if  line_point2[0] < point[0]:  #如果是左车道线
    #             line_p = (int(point[0]-distance * math.cos(vslope)/Wrate),int(point[1]+distance * math.sin(vslope)/Hrate))
    #         else:   #如果是右车道线
    #             line_p = (int(point[0]+distance * math.cos(vslope)/Wrate),int(point[1]-distance * math.sin(vslope)/Hrate))
    #         cv2.line(img,line_p,point,(180,180,180),2)
    #     else:
    #         line_point1 = (int(1980/2/Wrate),int(810/Hrate))
    #         line_point2 = (int(1980/2/Wrate),int(1080/Hrate))
    #     cv2.line(img,line_point1,line_point2,(105,200,255),2)
        #cv2.imwrite(os.path.join('test_output', 'result',str(idx)+'distance.jpg'), img)
    return distance



#计算车道线的P1 P2 P3 P4点坐标
def cal_lane_position(img,lines,p_position):
    left_lanes,right_lanes = [],[]
    #过滤操作，提取有用车道线
    if lines is None:
        return p_position
    for line in lines:
        lane = line[0]
        x1,y1,x2,y2 = lane[0],lane[1],lane[2],lane[3]
        slope = (y1-y2)/(x1-x2)
        # print(slope)
        if slope>-0.85 and slope<-0.55:
            cv2.line(img,(x1,y1),(x2,y2),255,1)
            left_lanes.append([(x1+x2)/2,(y1+y2)/2,slope])
        elif slope>0.53 and slope<0.89:
            cv2.line(img,(x1,y1),(x2,y2),255,1)
            right_lanes.append([(x1+x2)/2,(y1+y2)/2,slope])
        # else:
        #     print("slope:",slope)
    
    y_far = int(resize_height *810/1080)
    y_near = resize_height  

    # 根据斜率计算左车道线的P1 P3点
    if len(left_lanes):
        left_lanes = np.average(np.array(left_lanes),axis = 0).tolist()
        left_x_aver,left_y_aver,lslope = int(left_lanes[0]),int(left_lanes[1]),left_lanes[2]
        left_x_near = int(left_x_aver + (y_near - left_y_aver) / lslope)
        left_x_far = int(left_x_aver + (y_far - left_y_aver) / lslope)
    else:
        left_x_far = p_position[0][0]
        left_x_near = p_position[2][0]

    # 根据斜率计算右车道线的P2 P4点
    if len(right_lanes):
        right_lanes = np.average(np.array(right_lanes),axis = 0).tolist()
        right_x_aver,right_y_aver,rslope = int(right_lanes[0]),int(right_lanes[1]),right_lanes[2]
        right_x_near = int(right_x_aver + (y_near - right_y_aver) / rslope)
        right_x_far = int(right_x_aver + (y_far - right_y_aver) / rslope)
    else:
        right_x_far = p_position[1][0]
        right_x_near = p_position[3][0]

    #P1为左车道与y=192的交点；
    #P2为右车道与y=192的交点；
    #P3为左车道与y=256的交点；
    #P4为右车道与y=256的交点。
    p1 = (int(left_x_far) , int(y_far))
    p2 = (int(right_x_far) ,int(y_far))
    p3 = (int(left_x_near) , int(y_near))
    p4 = (int(right_x_near) , int(y_near))
    cv2.circle(img,p1,2,166,3)
    cv2.circle(img,p2,2,166,3)
    cv2.circle(img,p3,2,166,3)
    cv2.circle(img,p4,2,166,3)
    if is_visualize : cv2.imshow(' actual_line',img)
    return [p1,p2,p3,p4]

class lane_detector(object):
    def __init__(self):
        #图片尺寸相关
        self.origin_width = 1280
        self.origin_height = 720
        self.resize_width = 256
        self.resize_height = 256
        self.point = (int(self.resize_width/2)),int((192+self.resize_height)/2)#车道中心的点
        self.Wrate = self.origin_width/self.resize_width
        self.Hrate = self.origin_height/self.resize_height
        self.p_position = [(70,192),(190,192),(0,256),(256,256)]
        #差错控制
        self.kerror = 0.05
        self.derror = 10
        #图像处理
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        # mask_right_vec = np.array([[650,430],[1030,430],[1280,540],[1280,720],[980,720]])
        # mask_left_vec = np.array([[630,430],[250,430],[0,540],[0,720],[300,720]])
        # self.mask = np.zeros([self.origin_height,self.origin_width], dtype=np.uint8)
        # self.mask = cv2.fillConvexPoly(self.mask, mask_right_vec, 255)
        # self.mask = cv2.fillConvexPoly(self.mask, mask_left_vec, 255)
        mask_vec = np.array([[0,int(resize_height/2)],[resize_width,int(resize_height/2)],[resize_width,resize_height],[0,resize_height]])
        self.mask = np.zeros([resize_height,resize_width], dtype=np.uint8)
        cv2.fillConvexPoly(self.mask, mask_vec, 255)
        #显示相关
        self.doshow = True
        #计算相关
        self.actual_lines = []
        self.UnetTRT = UnetTRT()
        

    def detect(self,img):
        OutputImg = self.UnetTRT.unet_infer(img)
        # print("type:",type(OutputImg))
        # print("shape:",OutputImg.shape)

        lines = hough_lines(OutputImg,self.mask, rho=1, theta=np.pi / 180, threshold=10, min_line_len=10, max_line_gap=10)
        real_line = np.zeros([resize_height,resize_width], dtype=np.uint8)
        cv2.circle(real_line,self.point,2,166,3)
        self.p_position = cal_lane_position(real_line,lines,self.p_position)
        
        distance = get_distance_from_point_to_line(self.point,self.p_position)
        return OutputImg,distance
