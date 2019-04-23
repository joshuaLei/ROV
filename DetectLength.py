import cv2
import numpy as np
import math
import time
#from video import Video

class ROV():
    def __init__(self):
        self.video1 = cv2.VideoCapture(0)
        #self.video2 = Video(port=4777)
        self.srcframe = None
        self.frame = None
        self.mask = None

        self.areaval = 1000
        self.width = 1.7
        self.rect = 0
        self.box = None

        self.cal_val = 1

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
        hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        lower = np.array([0,0,0])
        upper = np.array([60,60,60])
        #lower = np.array([60,120,120])
        #upper = np.array([100,150,150])
        #lower = np.array([114,141,83])
        #upper = np.array([0,255,175])
        mask = cv2.inRange(hsv, lower, upper)
        self.mask = mask

    def detect(self):
        mask = self.mask
        frame = self.frame
        contours, hierarcy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                self.box = [box]
                #print()
                #return 0

    def calculation(self):
        x1 = self.box[0][0][0]
        x2 = self.box[0][1][0]
        y1 = self.box[0][0][1]
        y2 = self.box[0][1][1]
        C1 = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))

        x3 = self.box[0][1][0]
        x4 = self.box[0][2][0]
        y3 = self.box[0][1][1]
        y4 = self.box[0][2][1]
        C2 = math.sqrt(math.pow((x3 - x4), 2) + math.pow((y3 - y4), 2))

        if C1 < C2:
            ratio = abs(self.width/C1)
            cal = ratio*C2
        else:
            ratio = abs(self.width/C2)
            cal = ratio*C1

        #cal = cal + self.cal_val
        print('calculation',cal)
        cv2.putText(self.frame, 'cal = {}'.format(cal), (int(self.center[0][0]), int(self.center[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (215, 228, 41), 2)
        cv2.imshow('frame', self.frame)


if __name__ == "__main__":
    rov = ROV()
    i = 0
    #video = Video(port=4777)
    while True:
        #frame = rov.capture()
        cap = rov.debug()
        #if not video.frame_available():
            #continue

        #cap = video.frame()
        #frame = cap
        #rov.frame = cap
        #rov.srcframe = cap

        frame = cv2.resize(cap, (800, 600))
        rov.frame = frame
        k = cv2.waitKey(1)

        rov.msk()
        rov.detect()
        rov.calculation()
        #cv2.imshow('frame', rov.frame)


        if k == 32:
            cv2.imshow('freeze', frame)
            continue

        if k == ord('s'):
                cv2.imwrite('photos/image'+str(time.time())+ '.jpg', frame)
                print('save')
                i+=1

        if k == 27:
            print("stop")
            break

cv2.destroyAllWindows()