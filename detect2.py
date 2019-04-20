'''
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
def a(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sensitivity = 200
    lower_white = np.array([0, 0, 100])
    upper_white = np.array([131, 255, 255])

    # Threshold the HSV image to get only white colors
    thresh = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(frame,frame, mask= thresh)
    return thresh, res

while(1):
    ref, frame = cap.read()
    thresh, res = a(frame)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(1)

    #cv2.imshow('res',res)
    #cv2.waitKey(1)

cv2.destroyAllWindows()
'''
import cv2
import numpy as np
import math
import time
#from video import Video

def capture(cap):
    ret, frame = cap.read()
    img = mask(frame)
    return img


def mask(frame):
    frame = cv2.resize(frame, (800, 600))
    hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    lower = np.array([0,0,0])
    upper = np.array([60,60,60])
    #lower = np.array([60,120,120])
    #upper = np.array([100,150,150])
    #lower = np.array([114,141,83])
    #upper = np.array([0,255,175])
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('binary image', mask)
    img = detect(frame, mask, 1000)
    return img

def detect(frame, mask, areaval):
    contours, hierarcy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areaval:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            img = calculation(frame, box, 1.7, rect)
            return img
        #cv2.imshow('frame', frame)

def calculation(frame, box, width, center):
    x1 = box[0][0]
    x2 = box[1][0]
    y1 = box[0][1]
    y2 = box[1][1]
    C1 = math.sqrt(math.pow((x1 - x2),2) + math.pow((y1 - y2),2))

    x3 = box[1][0]
    x4 = box[2][0]
    y3 = box[1][1]
    y4 = box[2][1]
    C2 = math.sqrt(math.pow((x3 - x4),2) + math.pow((y3 - y4),2))

    if C1 < C2:
        ratio = abs(width/C1)
        cal = ratio*C2
    else:
        ratio = abs(width/C2)
        cal = ratio*C1

    print('calculation',cal)
    cv2.putText(frame, 'cal = {}'.format(cal), (int(center[0][0]), int(center[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (215, 228, 41), 2)
    cv2.imshow('frame', frame)
    return frame


if __name__ == "__main__":
    capt = cv2.VideoCapture(1)
    #video = Video(port=4777)
    i = 0

    while True:
        #if not video.frame_available():
            #continue

        #cap = video.frame()
        cap = capture(capt)
        img = cv2.resize(cap, (800, 600))

        frame = mask(img)

        k = cv2.waitKey(1)

        if k == 32:
            cv2.imshow('freeze', frame)
            continue

        if k == ord('s'):
                cv2.imwrite('photos/image'+str(time.time())+ '.jpg', cap)
                print('save')
                i += 1

        if k == 27:
            print("stop")
            break

cv2.destroyAllWindows()
#cap.release()


