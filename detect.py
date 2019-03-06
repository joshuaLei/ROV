import cv2
import numpy as np

class ROV:
    def __init__(self, lower, upper):
        self.lower = lower;
        self.upper = upper;

    def mask(self, frame):
        lower = self.lower;
        upper = self.upper;
        bgr = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(bgr, lower, upper)
        return mask

    def detection(self, mask, areaval=100,  lenval=0.03):
        triangle = 0
        square = 0
        circle = 0
        line = 0
        array = []
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > areaval:
                cnt_len = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, lenval * cnt_len, True)
                if len(approx) == 3:
                    if triangle >= 6:
                        triangle = 0
                    else:
                        triangle = triangle + 1
                elif len(approx) == 4:
                    if square >= 6:
                        square = 0
                    else:
                        square = square + 1
                elif ((len(approx) > 7) and (len(approx) < 9)):
                    if circle >= 6:
                        circle = 0
                    else:
                        circle = circle + 1
                        array.insert(0, [square, triangle, circle, line])
                else:
                    line = line + 1
        return array

    def export(self, square, triangle, circle, line):
        img = np.zeros((297, 210, 3), np.uint8)
        cv2.circle(img, (150, 40), 20, (0, 0, 255), -1)
        triangle_cnt = np.array([(150, 90), (130, 120), (170, 120)])
        cv2.line(img, (130, 180), (170, 180), (0, 0, 255), 4)
        cv2.rectangle(img, (170, 230), (130, 270), (0, 0, 255), -1)
        cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 255), -1)
        cv2.putText(img, str(circle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(triangle), (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(line), (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(square), (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for i in range(10):
            cv2.imwrite('ROV{}.png'.format(i), img)
            break

    def run(self, frame):
        m = self.mask(frame);
        cnt = self.detection(m);
        self.export(cnt[0][0], cnt[0][1], cnt[0][2], cnt[0][3])

frame = cv2.imread("img/shape.png")
mission = ROV([0,0,0], [70,70,70])
mission.run(frame)