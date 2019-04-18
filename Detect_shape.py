import cv2
import numpy as np
from time import sleep
#from video import Video
import time


class ROV:
    def __init__(self, debug=True):
        self._triangle = 0
        self._rectangle = 0
        self._circle = 0
        self._line = 0
        #self.debug = debug

        self._num_of_shapes = {
            "circle": 0,
            "line": 0,
            "triangle": 0,
            "rectangle": 0
        }

        self.srcframe = None
        self.mask = None
        self.cropped = None
        self.frame = None

        self.status = True

    def debug(self, frame):
        ret, frame = cap.read()
        img = cv2.resize(frame, (800, 600))
        self.srcframe = frame
        self.frame = img
        # print("capture")

    def capture(self, video):
        pass

    def preprocessing(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.mask = thresh
        # print("preprocess")
        print(thresh)
        return thresh

    def overlay(self, frame):
        overlay = frame.copy()
        cropped = frame.copy()
        img = frame[50:525, 75:750].copy()
        cv2.rectangle(overlay, (50, 75), (750, 525), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, cropped, 1 - 0.3, 0, cropped)
        self.frame = cropped
        self.cropped = img
        cv2.imshow('test', self.frame)
        cv2.waitKey(1)
        # print("overlay")

    def white_mask(self, image):
        thresh = self.preprocessing(self.cropped)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 1000
        max_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt

        if max_cnt is None:
            pass

        x, y, w, h = cv2.boundingRect(max_cnt)
        # print("white_mask")
        # self.mask = image[y:y + h, x:x + w].copy()

    def shape_mask(self, image):
        thresh = self.preprocessing(self.cropped)
        thresh = 255 - thresh
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_h, image_w, _ = image.shape
        bounding = 0.02
        bounding_x = image_w * bounding
        bounding_y = image_h * bounding

        target_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                x, y, w, h = cv2.boundingRect(cnt)
                if x > bounding_x and x + w < image_w - bounding_x and \
                        y > bounding_y and y + h < image_h - bounding_y:
                    target_contours.append(cnt)

        min_x = image_w
        min_y = image_h
        max_x = 0
        max_y = 0

        for cnt in target_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if x + w > max_x:
                max_x = x + w
            if x < min_x:
                min_x = x
            if y + h > max_y:
                max_y = y + h
            if y < min_y:
                min_y = y

        space = 10
        min_y = max(min_y - space, 0)
        max_y = min(max_y + space, image_h)
        min_x = max(min_x - space, 0)
        max_x = min(max_x + space, image_w)

        # cv2.imshow("", image[min_y:max_y, min_x:max_x])
        # cv2.waitKey(1)

        self.mask = image[min_y:max_y, min_x:max_x].copy()
        # print("shape_mask")

    def detection(self, image, is_draw=False):
        if image.shape[0] == 0 or image.shape[1] == 0:
            self.status = False
            return [0, 0]
        else:

            self._num_of_shapes["circle"] = 0
            self._num_of_shapes["line"] = 0
            self._num_of_shapes["triangle"] = 0
            self._num_of_shapes["rectangle"] = 0

            org_h, org_w, _ = image.shape
            rate_x = 800 / org_w
            rate_y = 450 / org_h
            image = cv2.resize(image, (800, 450), interpolation=cv2.INTER_AREA)
            #thresh = self.preprocessing(self.cropped)
            mask = cv2.cvtColor(self.cropped, cv2.COLOR_HSV2BGR)
            lower = np.array([0, 0, 0])
            upper = np.array([70, 70, 70])
            mask = cv2.inRange(mask, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            active_cnts = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    epsilon = 0.03 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    vertex = 0
                    for i in range(len(approx)):
                        p1 = approx[i]
                        p2 = approx[(i + 1) % len(approx)]
                        e = np.sqrt(np.sum(abs(p1 - p2) ** 2))
                        if e >= 25:
                            vertex += 1
                    # vertex = len(approx)

                    # print(cnt[0], cnt[1], "...")
                    active_cnts.append(cnt)
                    if is_draw:
                        cv2.drawContours(image, [cnt], 0, (0, 255, 0), 1)
                        cv2.putText(image, str(vertex), (x + int(w / 2) - 5, y + int(h / 2) + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)

                    if vertex == 2:
                        if self._num_of_shapes["line"] >= 6:
                            self._num_of_shapes["line"] = 0
                        else:
                            self._num_of_shapes["line"] += 1
                    elif vertex == 3:
                        if self._num_of_shapes["triangle"] >= 6:
                            self._num_of_shapes["triangle"] = 0
                        else:
                            self._num_of_shapes["triangle"] += 1
                    elif vertex == 4:
                        (x, y, w, h) = cv2.boundingRect(approx)
                        ar = w / float(h)
                        if ar >= 0.5 and ar <= 1.5:
                            if self._num_of_shapes["rectangle"] >= 6:
                                self._num_of_shapes["rectangle"] = 0
                            else:
                                self._num_of_shapes["rectangle"] += 1
                        else:
                            if self._num_of_shapes["line"] >= 6:
                                self._num_of_shapes["line"] = 0
                            else:
                                self._num_of_shapes["line"] += 1
                    else:
                        if self._num_of_shapes["circle"] >= 6:
                            self._num_of_shapes["circle"] = 0
                        else:
                            self._num_of_shapes["circle"] += 1
        self.status = True
        return active_cnts

    def show(self, img):
        cv2.circle(img, (150, 40), 20, (0, 0, 255), -1)
        triangle_cnt = np.array([(150, 90), (130, 120), (170, 120)])
        cv2.line(img, (130, 180), (170, 180), (0, 0, 255), 4)
        cv2.rectangle(img, (170, 230), (130, 270), (0, 0, 255), -1)
        cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 255), -1)
        cv2.putText(img, str(self._circle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(self._triangle), (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(self._line), (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(self._rectangle), (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.frame = img


if __name__ == "__main__":
    rov = ROV(True)

    cap = cv2.VideoCapture(0)
    #video = Video(port=4777)

    i = 0

    while True:
        #if not video.frame_available():
            #continue

        #cap = video.frame()
        #img = cv2.resize(cap, (800, 600))
        #rov.srcframe = cap
        #rov.frame = img

        k = cv2.waitKey(1)

        rov.debug(cap)
        # rov.capture(video)
        rov.overlay(rov.frame)
        rov.preprocessing(rov.cropped)
        rov.white_mask(rov.cropped)
        rov.shape_mask(rov.cropped)
        active_cnts = rov.detection(rov.mask, False)
        if rov.status == False:
            print("none")
            #rov.status = True
            continue
        else:
            cv2.drawContours(rov.frame, active_cnts, -1, (0, 255, 0), 2)
            rov.show(rov.frame)
            cv2.imshow('frame', rov.frame)

        if k == 32:
            rov._circle = rov._num_of_shapes["circle"]
            rov._line = rov._num_of_shapes["line"]
            rov._triangle = rov._num_of_shapes["triangle"]
            rov._rectangle = rov._num_of_shapes["rectangle"]

        if k == 27:
            print("stop")
            break

        if k == ord('s'):
            cv2.imwrite('photos/image' + str(time.time()) + '.jpg', rov.source_img)
            print('save')
            i += 1

        if rov.frame is None:
            continue

    cv2.destroyAllWindows()
    # cap.release()
