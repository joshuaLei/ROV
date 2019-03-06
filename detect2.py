import cv2
import numpy as np
import math

class ROV:
    def __init__(self):
        pass

    def mask(self, frame):
        bgr = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        lower = np.array([0, 0, 0])
        upper = np.array([70, 70, 70])
        mask = cv2.inRange(bgr, lower, upper)
        return mask


    def detect(self, frame, mask, areaval=10000):
        im2, contours, hierarcy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > areaval:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                cv2.imshow('frame', frame)
                self.calculation(frame, box, 1.8, rect)

    def calculation(self, frame, box, width, center):
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
            ratio = abs(width / C1)
            cal = ratio * C2
        else:
            ratio = abs(width / C2)
            cal = ratio * C1

        print('calculation', cal)
        cv2.putText(frame, 'cal = {}'.format(cal), (int(center[0][0]), int(center[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (215, 228, 41), 2)
        cv2.imshow('frame', frame)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        task = ROV()
        task.mask(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("error")

cv2.destroyAllWindows()
cap.release()