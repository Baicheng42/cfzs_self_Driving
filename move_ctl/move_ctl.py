#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import serial
import numpy as np
import string
# from _typeshed import Self

class PID(object):
    def __init__(
        self,
        Kp=0.2,
        Ki=0.0,
        Kd=0.1,
        seterror=0
    ):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.error = 0
        self.lasterror = seterror
        self.integration = 0
        self.out = 0
        self.out_past = 0

    def seterror(self, error):
        # onestep = 10
        self.lasterror = self.error
        self.error = error
        self.integration = self.integration + error
        if self.integration > 1000:
            self.integration = 1000
        elif self.integration < -1000:
            self.integration = -1000
        self.out_past = self.out
        self.out = self.Kp * self.error + self.Ki * self.integration - self.Kd*(self.error - self.lasterror)
        # if self.out - self.out_past >onestep:
        #     self.out = self.out_past + onestep
        # elif self.out - self.out_past < (0-onestep):
        #     self.out = self.out_past - onestep
        # if abs(self.out - self.out_past) > onestep:
        #     self.out = self.out_past

class move_ctl(object):
    def __init__(self, sym32_port = '/dev/ttyUSB0'):
        self.ser = serial.Serial(sym32_port, 115200, timeout=5)
        self.pid_ctler = PID()
        speed = 0
        steering_angle = 0
        target_speed = abs(speed) * 32 / 0.41
        target_angle = steering_angle + 60
        if target_angle < 25:
            target_angle = 25
        elif target_angle > 95:
            target_angle = 95
        if not self.ser.is_open:
            print('串口打开失败')
            exit(1)
        sum = (0xa5 + 0x5a + 0x06 + target_angle + target_speed) % 256
        send_msg = chr(0xa5)+chr(0x5a)+chr(0x06)+chr(0x00)+chr(int(target_angle))+chr(int(target_speed))+chr(0x00)+chr(0x00)+chr(0x00)+chr(int(sum))
        self.ser.write(send_msg.encode())

    def __del__(self):
        self.send(send_speed = 0, steering_angle = 0)

    def send(self, send_speed = 0, steering_angle = 0):
        target_speed = abs(send_speed) * 32 / 0.41
        target_angle = steering_angle + 60
        if target_angle < 25:
            target_angle = 25
        elif target_angle > 95:
            target_angle = 95
        sum = (0xa5 + 0x5a + 0x06 + target_angle + target_speed) % 256
        send_msg = chr(0xa5)+chr(0x5a)+chr(0x06)+chr(0x00)+chr(int(target_angle))+chr(int(target_speed))+chr(0x00)+chr(0x00)+chr(0x00)+chr(int(sum))
        self.ser.write(send_msg.encode())
    
    def pid_ctl(self, position, speed = 0.1):
        # error = (position - 640)/ 640
        self.pid_ctler.seterror(position)
        print("angle:", self.pid_ctler.out)
        return self.pid_ctler.out
        # self.send(send_speed = speed, steering_angle = self.pid_ctler.out)