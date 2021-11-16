import datetime
from move_ctl import move_ctl
import numpy as np
import cv2


categories = ["go_straight", "go_and_right", "turn_left", "left_go_right", "zebra", "people", "obstacle", "traffic_light","end"]

class state_machine(object):
    def __init__(self):
        self.test = True
        self.is_end = False
        self.tempt = 10
        #output
        self.speed = 0
        self.angle = 0

        #时间相关
        self.label_zhi = 2.5
        self.label_xun = 4
        self.label_turn = 4.7
        self.obstacle_zhuan = 1.65
        self.obstacle_hui = 2.2
        self.zebra_time = 5
        self.people_time = 3
        self.end_time = 4.2

        self.move_ctl = move_ctl()

        #0：正常寻迹    1：有行人   2：有红绿灯   3：有转弯标志 4：有障碍物 5：有斑马线
        self.state = 0
        self.state_people = 0
        self.state_light = 0
        self.state_label = 0
        self.state_obstacle = 0
        self.state_zebra = 0
        self.state_end = 0

        self.in_lane_dis = 60
        self.width = 256
        self.heigh = 256
        self._score = 0.001
        self.people_dis = 130
        self.light_dis = 130#################
        self.label_dis = 180
        self.obstacle_dis = 160
        self.zebra_dis = 200
        self.end_dis = 160

        self.obj_x = np.full(9, -1)
        self.obj_y = np.full(9, -1)
        self.is_red = False
    
    def run(self, img, distence, result_boxes, result_scores, result_classid):
        while self.is_end:
            self.move_ctl.send(send_speed = 0, steering_angle = 0)
        self.change_state(img, result_boxes, result_scores, result_classid)
        self.do_action(distence)
        print('or_speed = ', self.speed, ' or_angle = ', self.angle)
        if self.test:
            self.speed = 0
            self.angle = 0
            print('send_speed = ', self.speed, ' steering_angle = ', self.angle)
        self.move_ctl.send(send_speed = self.speed, steering_angle = self.angle)

    def traffic_condition(self, img, box):
        x1, y1, x2, y2 = box
        xmin = int(min(x1, x2))
        xmax = int(max(x1, x2))
        ymin = int(min(y1, y2))
        ymax = int(max(y1, y2))
        dy = int((ymax - ymin)/4)
        dx = int((xmax - xmin)/12)
        red_sub_img = img[ymin+dy:ymax-dy,xmin+dx:xmin+int((xmax-xmin)/3)-dx]
        yellow_sub_img = img[ymin+dy:ymax-dy,xmin+int((xmax-xmin)/3)+dx:xmin+int((xmax-xmin)*2/3)-dx]
        green_sub_img = img[ymin+dy:ymax-dy,xmin+int((xmax-xmin)*2/3)+dx:xmax-dx]
        green_l = cv2.cvtColor(green_sub_img, cv2.COLOR_BGR2GRAY).mean()
        red_l = cv2.cvtColor(red_sub_img, cv2.COLOR_BGR2GRAY).mean()
        yellow_l = cv2.cvtColor(yellow_sub_img, cv2.COLOR_BGR2GRAY).mean()
        traffic_red = True
        if green_l<110:
            traffic_red = True
        else:
            traffic_red = False
        return traffic_red
        #return green_l - max(red_l, yellow_l) < self.tempt

    def change_state(self, img, result_boxes, result_scores, result_classid):
        #识别出所有的符合要求的物品
        obj_len = len(result_boxes)
        self.obj_x = np.full(9, -1)
        self.obj_y = np.full(9, -1)
        # self.is_red = False
        for i in range(obj_len):
            box = result_boxes[i]
            x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
            locate_x = (x1 + x2)/2
            locate_y = max(y1,y2)
            score = result_scores[i]
            id = result_classid[i]
            if id == 6.0:
                if not self.in_lane(locate_x, locate_y):
                    continue
            if score >self._score:
                self.obj_x[int(id)] = locate_x
                self.obj_y[int(id)] = locate_y
                if id == 7.0:
                    self.is_red = self.traffic_condition(img, box)

        #对所有物品进行判断并进入相应的state
        if self.state == 0:
            self.state_people = 0
            self.state_light = 0
            self.state_label = 0
            self.state_obstacle = 0
            self.state_zebra = 0
            self.state_end = 0
            #行人
            if self.obj_y[5] > self.people_dis:
                self.state = 1
            #红绿灯
            elif self.obj_y[7] < self.light_dis and self.obj_y[7] > 0:
                self.state = 2
            #转弯标识
            elif self.obj_y[1] > self.label_dis:
                self.state = 3
            #障碍物
            elif self.obj_y[6] > self.obstacle_dis:
                self.state = 4
            #斑马线
            elif self.obj_y[4] > self.zebra_dis:
                self.state = 5
            #结束
            elif self.obj_y[8] > self.end_dis:
                self.state = 6
            else:
                self.state = 0

        if self.obj_y[5] > self.people_dis:
            self.state = 1
        elif self.obj_y[7] < self.light_dis and self.obj_y[7] > 0:
            self.state = 2

    def do_action(self, distence):
        if self.state == 0:
            self.state = self.action_nomal(distence)
        elif self.state == 1:
            self.state = self.action_people(distence)
        elif self.state == 2:
            self.state = self.action_light(distence)
        elif self.state == 3:
            self.state = self.action_label(distence)
        elif self.state == 4:
            self.state = self.action_obstacle(distence)
        elif self.state == 5:
            self.state = self.action_zebra(distence)
        elif self.state == 6:
            self.state = self.action_end(distence)
        else:
            self.state = 0
        
    def action_nomal(self, distence):#寻迹
        self.angle = self.move_ctl.pid_ctl(distence)
        self.speed = 0.4
        return 0

    def action_people(self, distence):#有人5
        if self.state_people == 0:
            self.angle = self.move_ctl.pid_ctl(distence)
            self.speed = 0.2
            self.state_people = 1
            return 1
        elif self.state_people == 1:
            if self.obj_y[5] > 0:
                self.start_time = datetime.datetime.now()
                self.state_people = 2
                self.people_count = 0
                self.angle = 0
                self.speed = 0
                return 1
            return 0
        elif self.state_people == 2:
            if self.obj_y[5] > 0:
                self.start_time = datetime.datetime.now()
            self.now_time = datetime.datetime.now()
            temp = (self.now_time-self.start_time)
            time = temp.microseconds/1000000 + temp.seconds
            if time > self.people_time:
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            else:
                self.angle = 0
                self.speed = 0
                return self.state

    def action_light(self, distence):#有红绿灯7
        if self.state_light == 0:
            if self.is_red and self.obj_y[7] > 0:#存在且为红灯
                self.state_light = 1
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.4
                return self.state
            else:#不存在或为绿灯，退出
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.4
                return 0
        elif self.state_light == 1:
            if self.is_red and self.obj_y[7] > 0:#存在且为红灯:
                self.angle = 0
                self.speed = 0
                self.state_light = 2
                self.light_start_time = datetime.datetime.now()
                return self.state
            else:#不存在或为绿灯，退出
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
        elif self.state_light == 2:
            if self.is_red and self.obj_y[7] > 0:#存在且为红灯，则将时间重置
                self.angle = 0
                self.speed = 0
                self.light_start_time = datetime.datetime.now()
            self.light_now_time = datetime.datetime.now()
            temp = (self.light_now_time-self.light_start_time)
            time = temp.microseconds/1000000 + temp.seconds
            if time > 3:#3s未识别到红灯
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            return self.state
        return 0

    def action_label(self, distence):#有转弯标签1
        if self.state_label == 0:
            if self.obj_y[1] < 0:#未识别到
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            elif self.obj_y[1] < self.label_dis:#不在范围内
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return self.state
            else:#识别到且在范围内
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                self.state_label = 1
                return self.state
        elif self.state_label == 1:
            if self.obj_y[1] < 0:#第二次未识别到
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            elif self.obj_y[1] < self.label_dis:#第二次识别到但不在范围内
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                self.state_label = 0
                return self.state
            else:#第二次识别到且在范围内
                self.label_start_time = datetime.datetime.now()
                self.state_label = 2
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return self.state

        elif self.state_label == 2:
            #寻迹7.3秒，若在该段时间内未检测到障碍，则直行，检测到障碍则进入转弯模式
            self.label_now_time = datetime.datetime.now()
            temp = (self.label_now_time-self.label_start_time)
            time = temp.microseconds/1000000 + temp.seconds
            if time > (self.label_zhi + self.label_xun):#超过7.3s退出
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            elif time > self.label_xun:#超过3s直行
                self.angle = 1
                self.speed = 0.2
            else:#3s以内寻迹
                self.angle = self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
            if self.obj_y[6] > 0:#检测到障碍进入转弯模式
                self.state_label = 3
            return self.state

        elif self.state_label == 3:
            #继续寻迹若干时间
            self.label_now_time = datetime.datetime.now()
            temp = (self.label_now_time-self.label_start_time)
            time = temp.microseconds/1000000 + temp.seconds
            if time > (self.label_zhi + self.label_xun):#超过7.3s转弯
                self.angle = -35
                self.speed = 0.2
                self.label_start_time = datetime.datetime.now()
                self.state_label = 4
            elif time > self.label_xun:#超过3s直行
                self.angle = 1
                self.speed = 0.2
            else:#3s以内寻迹
                self.angle = self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
            return self.state

        elif self.state_label == 4:
            #转弯5s
            self.label_now_time = datetime.datetime.now()
            temp = (self.label_now_time-self.label_start_time)
            time = temp.microseconds/1000000 + temp.seconds
            if time > self.label_turn:
                #到5s后退出label状态
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            else:
                self.angle = -35
                self.speed = 0.2
            return self.state

    def action_obstacle(self, distence):#有障碍物6
        if self.state_obstacle == 0:
            if self.obj_y[6] > self.obstacle_dis:
                self.state_obstacle = 1
                return 4
            else:
                self.state_obstacle = 0
                return 0
        elif self.state_obstacle == 1:
            if self.obj_y[6] > self.obstacle_dis:
                self.state_obstacle = 2
                self.obstacle_start_time = datetime.datetime.now()
                self.speed = 0.3
                self.angle = -30
                return 4
            else:
                self.state_obstacle = 0
                return 0
        #识别到障碍物
        elif self.state_obstacle == 2:
            self.obstacle_now_time = datetime.datetime.now()
            temp = (self.obstacle_now_time-self.obstacle_start_time)
            time = temp.microseconds/1000000 + temp.seconds
            if time > self.obstacle_zhuan:
                self.obstacle_start_time = datetime.datetime.now()
                self.state_obstacle = 3
                self.speed = 0.3
                self.angle = 30
            return 4
        elif self.state_obstacle == 3:
            self.obstacle_now_time = datetime.datetime.now()
            temp = (self.obstacle_now_time-self.obstacle_start_time)
            time = temp.microseconds/1000000 + temp.seconds
            if time > self.obstacle_hui:
                self.state_obstacle = 0
                self.speed = 0.3
                self.angle = 0
                return 0
            return 4

    def action_zebra(self, distence):#有斑马线4
        # self.angle = self.move_ctl.pid_ctl(distence)
        if self.state_zebra == 0:
            if self.obj_y[4] > self.zebra_dis:
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.4
                self.state_zebra = 1
            elif self.obj_y[4] > 0:
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
            else:
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            return self.state
        
        elif self.state_zebra == 1:
            if self.obj_y[4] > self.zebra_dis:
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                self.start_time = datetime.datetime.now()
                self.state_zebra = 2
            elif self.obj_y[4] > 0:
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
            else:
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            return self.state
        #确认两次存在斑马线之后
        elif self.state_zebra == 2:
            self.now_time = datetime.datetime.now()
            temp = (self.now_time-self.start_time)
            time = temp.microseconds/1000000 + temp.seconds
            if time > self.zebra_time:
                self.angle = self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            else:
                self.angle = 8
                self.speed = 0.2
                return self.state

        elif self.state_zebra == 3:#双斑马线路口
            self.now_time = datetime.datetime.now()
            temp = (self.now_time-self.start_time)
            time = temp.microseconds/1000000 + temp.seconds
            if time > 2:
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            else:
                self.angle = 3
                self.speed = 0.2
                return self.state

    def action_end(self, distence):
        if self.state_end == 0:
            if self.obj_y[8] > self.end_dis:
                self.state_end = 1
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return self.state
            else:
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
        elif self.state_end == 1:
            if self.obj_y[8] > self.end_dis:
                self.state_end = 2
                self.start_time = datetime.datetime.now()
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return self.state
            else:
                self.angle = self.move_ctl.pid_ctl(distence)
                self.speed = 0.2
                return 0
            
        elif self.state_end == 2:#直行一段时间
            self.now_time = datetime.datetime.now()
            temp = (self.now_time-self.start_time)
            time = temp.microseconds/1000000 + temp.seconds
            if time > self.end_time:#停
                self.angle = 0
                self.speed = 0
                self.is_end = True
                return 0
            elif time > (self.end_time/2):#后1/2段直行
                # self.angle = self.move_ctl.pid_ctl(distence)
                self.angle = 5
                self.speed = 0.2
                return self.state
            else:#前1/2段寻迹
                self.angle = self.move_ctl.pid_ctl(distence)
                # self.angle = 0
                self.speed = 0.2
                return self.state

    def in_lane(self,locate_x, locate_y):
        flag1 = (locate_x > self.width/2-self.in_lane_dis)and(locate_x < self.width/2+self.in_lane_dis)
        return flag1
