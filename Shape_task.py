import cv2
import numpy as np
#from video import Video
import time
from PIL import Image, ImageEnhance


class ROV:
    def __init__(self, debug=True):
        self.video1 = cv2.VideoCapture(0)

        self._triangle = 0
        self._rectangle = 0
        self._circle = 0
        self._line = 0

        self.min_x = 0
        self.min_y = 0

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
        self.areaval = 1000

    def debug(self):
        success, self.frame = self.video1.read()
        self.frame = cv2.flip(self.frame, 3)
        self.srcframe = self.frame
        return self.frame

    def msk(self, image):
        if image.shape[0] == 0 or image.shape[1] == 0:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.medianBlur(image, 5)
        lower = np.array([0, 0, 150])
        upper = np.array([255, 100, 255])
        mask = cv2.inRange(image, lower, upper)
        self.mask = mask
        return mask

    def preprocessing(self, image):
        image = ImageEnhance.Color(Image.fromarray(image)).enhance(1.1)
        image = ImageEnhance.Brightness(image).enhance(1.2)
        image = ImageEnhance.Contrast(image).enhance(1.2)
        image = ImageEnhance.Sharpness(image).enhance(1.1)
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.erode(np.array(image), kernel, iterations=1)
        image = cv2.dilate(np.array(image), kernel, iterations=1)
        return image

    def overlay(self, frame):
        overlay = frame.copy()
        cropped = frame.copy()
        img = frame[180:690, 160:600].copy()
        self.cropped = img
        self.srcframe = cropped

    def white_mask(self):
        thresh = self.msk(self.cropped)
        if thresh is None:
            return None
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 100
        max_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt

        if max_cnt is None:
            return None

        x, y, w, h = cv2.boundingRect(max_cnt)

        self.mask = self.cropped[y:y + h, x:x + w].copy()
        cropped = self.cropped[y:y + h, x:x + w].copy()
        return cropped, x, y, w, h

    def shape_mask(self):
        thresh = self.msk(self.mask)
        if thresh is None:
            return None
        thresh = 255 - thresh
        image = self.mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_h, image_w = image.shape
        bounding = 0.02
        bounding_x = image_w * bounding
        bounding_y = image_h * bounding

        target_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 150:
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

        x = min_x
        y = min_y

        self.max_x = max_x
        self.max_y = max_y

        w = max_x - min_x
        h = max_y - min_y

        self.mask = image[min_y:max_y, min_x:max_x].copy()
        cropped = self.cropped[min_y:max_y, min_x:max_x].copy()
        return cropped, x, y, w, h

    def detection(self, image, is_draw=True):
        if image.shape[0] == 0 or image.shape[1] == 0:
            return [0, 0]
        else:
            self._num_of_shapes["circle"] = 0
            self._num_of_shapes["line"] = 0
            self._num_of_shapes["triangle"] = 0
            self._num_of_shapes["rectangle"] = 0
            cv2.imshow('000', image)
            image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
            cropped = image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(image)
            mask = cv2.inRange(equ, 0, 52)
            cv2.imshow("equ", cv2.inRange(equ, 0, 52))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_h, image_w = mask.shape
            bounding = 0.015
            bounding_x = image_w * bounding
            bounding_y = image_h * bounding
            active_cnts = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.areaval:
                    x, y, w, h = cv2.boundingRect(cnt)
                    epsilon = 0.03 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    if x > bounding_x and x + w < image_w - bounding_x and \
                            y > bounding_y and y + h < image_h - bounding_y:
                        vertex = 0
                        for i in range(len(approx)):
                            p1 = approx[i]
                            p2 = approx[(i + 1) % len(approx)]
                            e = np.sqrt(np.sum(abs(p1 - p2) ** 2))
                            if e >= 25:
                                vertex += 1
                                active_cnts.append(cnt)

                        cv2.drawContours(cropped, [cnt], 0, (0, 255, 0), 3)
                        cv2.putText(cropped, str(vertex), (x + int(w / 2) - 5, y + int(h / 2) + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), lineType=cv2.LINE_AA)
                        #cv2.imshow('numbers', self.frame)

                        area = cv2.contourArea(cnt)

                        if area > 3000:
                            if vertex == 2:
                                if self._num_of_shapes["line"] >= 6:
                                    self._num_of_shapes["line"] = 6
                                else:
                                    self._num_of_shapes["line"] += 1

                            elif vertex == 3:
                                if self._num_of_shapes["triangle"] >= 6:
                                    self._num_of_shapes["triangle"] = 6
                                else:
                                    self._num_of_shapes["triangle"] += 1

                            elif vertex == 4:
                                (x, y, w, h) = cv2.boundingRect(approx)
                                ar = w / float(h)
                                if ar >= 0.2 and ar <= 1.8:
                                    if self._num_of_shapes["rectangle"] >= 6:
                                        self._num_of_shapes["rectangle"] = 6
                                    else:
                                        self._num_of_shapes["rectangle"] += 1
                                else:
                                    if self._num_of_shapes["line"] >= 6:
                                        self._num_of_shapes["line"] = 6
                                    else:
                                        self._num_of_shapes["line"] += 1

                            elif vertex == 0 or vertex == 1 or vertex == 5 or vertex == 6 \
                                    or vertex == 7 or vertex == 8:
                                if self._num_of_shapes["circle"] >= 6:
                                    self._num_of_shapes["circle"] = 6
                                else:
                                    self._num_of_shapes["circle"] += 1
                            else:
                                pass
                        else:
                            if self._num_of_shapes["line"] >= 6:
                                self._num_of_shapes["line"] = 6
                            else:
                                self._num_of_shapes["line"] += 1
        return cropped

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
    #video = Video(port=4777)
    i = 0

    while True:
        k = cv2.waitKey(1)
        #if not video.frame_available():
            #continue
        #img = video.frame()
        img = rov.debug()
        # img = cv2.imread('photos/train2/img0.jpg')
        rov.srcframe = img
        cap = rov.preprocessing(img)
        frame = cv2.resize(cap, (800, 600))
        rov.srcframe = frame
        rov.cropped = frame
        rov.overlay(frame)
        rov.msk(rov.cropped)
        img, x1, y1, w1, h1 = rov.white_mask()
        img2, x2, y2, w2, h2 = rov.shape_mask()
        frame = rov.detection(img2, False)
        try:
            src =  rov.srcframe
            overlay = src.copy()
            cv2.circle(src, (x + 160, y + 180), 10, (255, 0, 0), -1)
            cv2.circle(src, (x + w + 160, y + 180), 10, (255, 0, 0), -1)
            cv2.circle(src, (x + 160, y + h + 180), 10, (255, 0, 0), -1)
            cv2.circle(src, (x + w + 160, y + h + 180), 10, (255, 0, 0), -1)
            frame = cv2.resize(frame, (w, h))
            src[y + 180:y + h + 180, x + 160:x + w + 160] = frame
            rov.show(src)
            color = src.copy()
            cv2.rectangle(src, (180, 160), (690, 600), (0, 0, 255), -1)
            color = cv2.addWeighted(src, 0.4, color, 1 - 0.4, 0)
            cv2.imshow('frame', color)
        except Exception:
            print('error')

        if k == 32:
            rov._circle = rov._num_of_shapes["circle"]
            rov._line = rov._num_of_shapes["line"]
            rov._triangle = rov._num_of_shapes["triangle"]
            rov._rectangle = rov._num_of_shapes["rectangle"]

        if k == 27:
            print("stop")
            break

        if k == ord('s'):
            cv2.imwrite('photos/img' + str(time.time()) + '.jpg', frame)
            print('saved')
            i += 1

    cv2.destroyAllWindows()