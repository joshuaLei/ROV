import cv2
import numpy as np
import math
import time
from video import Video

class ROV():
    def __init__(self):
        self.video1 = cv2.VideoCapture(0)
        #self.video2 = Video(port=4777)
        self.srcframe = None
        self.frame = None
        self.mask = None

        self.areaval = 1300
        self.width = 1.7
        self.rect = 0
        self.box = None

        self.cal_val = 3

    def capture(self):
        '''
        success, self.frame = self.video2.read()
        self.frame = cv2.flip(self.frame, 3)
        self.srcframe = cv2.flip(self.frame, 3)
        return self.frame
        '''
        
        video = self.video2
        if not video.frame_available():
            return None
        cap = video.frame()
        self.frame = cap
        self.srcframe = cap
        return self.frame


    def debug(self):
        success, self.frame = self.video1.read()
        self.frame = cv2.flip(self.frame, 3)
        self.srcframe = cv2.flip(self.frame, 3)
        return self.frame

    def msk(self):
        frame = self.frame
        cv2.imshow('frame99',frame)
        kernel = np.ones((12, 12), np.uint8)
        frame = cv2.dilate(frame, kernel, iterations=1)
        #frame = cv2.erode(frame, kernel, iterations=1)
        self.frame = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        lower = np.array([0,0,0])
        upper = np.array([80,80,80])
        #lower = np.array([60,120,120])
        #upper = np.array([100,150,150])
        #lower = np.array([114,141,83])
        #upper = np.array([0,255,175])
        mask = cv2.inRange(hsv, lower, upper)
        self.mask = mask
        cv2.imshow('mask', mask)

    def detect(self):
        mask = self.mask
        frame = self.frame
        _, contours, hierarcy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.areaval:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                self.center = rect
                self.frame = frame
                self.rect = rect
                #self.box = [box]
                self.calculation(box)
                #print()
                #return 0

    def calculation(self, box):
        x1 = box[0][0]
        x2 = box[1][0]
        y1 = box[0][1]
        y2 = box[1][1]
        C1 = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))

        x3 = box[1][0]
        x4 = box[2][0]
        y3 = box[1][1]
        y4 = box[2][1]
        C2 = math.sqrt(math.pow((x3 - x4), 2) + math.pow((y3 - y4), 2))

        if C1 < C2:
            ratio = abs(self.width/C1)
            cal = ratio*C2
        else:
            ratio = abs(self.width/C2)
            cal = ratio*C1

        #print(ratio)
        '''
        if cal > 12 and cal < 15:
            self.cal_val = 1
        elif cal > 15 and cal < 19:
            self.cal_val = 1.5
        elif cal > 19 and cal < 23:
            self.cal_val = 2
        elif cal > 23 and cal < 28:
            self.cal_val = 2.5
        '''
        if cal > 25 and cal < 28:
            self.cal_val = 3
        elif cal > 28 and cal < 32:
            self.cal_val = 3.5
        elif cal > 32 and cal < 35:
            self.cal_val = 4
        cal = cal - self.cal_val

        #print('calculation',cal)
        cv2.putText(self.frame, 'cal = {}'.format(cal), (int(self.center[0][0]), int(self.center[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (215, 228, 41), 2)
        cv2.imshow('frame', self.frame)


if __name__ == "__main__":
    rov = ROV()
    i = 0
    video = Video(port=4777)
    while True:
        #frame = rov.capture()
        #cap = rov.debug()
        if not video.frame_available():
            continue

        cap = video.frame()
        frame = cap
        rov.frame = cap
        rov.srcframe = cap

        frame = cv2.resize(cap, (800, 600))
        rov.frame = frame
        k = cv2.waitKey(1)

        rov.msk()
        rov.detect()
        #rov.calculation()
        #cv2.imshow('frame', rov.frame)


        if k == 32:
            cv2.imshow('freeze', rov.frame)
            continue

        if k == ord('s'):
                cv2.imwrite('photos/image'+str(time.time())+ '.jpg', rov.frame)
                print('save')
                i+=1

        if k == 27:
            print("stop")
            break

cv2.destroyAllWindows()