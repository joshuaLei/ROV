import cv2
import numpy as np
import math

_triangle = 0
_rectangle = 0
_circle = 0
_line = 0

_num_of_shapes = {
    "circle": 0,
    "line": 0,
    "triangle": 0,
    "rectangle": 0
}

class ROV:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def preprocessing(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('thresh', thresh)
        cv2.waitKey(1)
        return thresh

    def white_mask(self, image):
        thresh = self.preprocessing(image)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt

        if max_cnt is None:
            return None

        x, y, w, h = cv2.boundingRect(max_cnt)
        return image[y:y + h, x:x + w].copy()

    def shape_mask(self, image):
        thresh = self.preprocessing(image)
        thresh = 255 - thresh
        a, contours, b = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_h, image_w, _ = image.shape
        bounding = 0.01
        bounding_x = image_w * bounding
        bounding_y = image_h * bounding

        target_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                if x > bounding_x and x + w < image_w - bounding_x and \
                        y > bounding_y and y + h < image_h - bounding_y:
                    target_contours.append(cnt)

                cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)

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

        space = 0
        min_y = max(min_y - space, 0)
        max_y = min(max_y + space, image_h)
        min_x = max(min_x - space, 0)
        max_x = min(max_x + space, image_w)

        return image[min_y:max_y, min_x:max_x].copy()

    def detection(self, image, is_draw=False):
        if image is None:
            print('none')
            return None, 0, 0, 0, 0
        else:
            print(image.shape)
            if image.shape[0] == 0 or image.shape[1] == 0:
                return None, 0, 0, 0, 0
            cv2.imshow('image', image)
            cv2.waitKey(1)
            image = cv2.resize(image, (800, 450), interpolation = cv2.INTER_AREA)
            thresh = self.preprocessing(image)
            thresh = 255 - thresh
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                        if e >= 15:
                            vertex += 1
                    # vertex = len(approx)

                    if is_draw:
                        cv2.drawContours(image, [cnt], 0, (0, 255, 0), 1)
                        cv2.putText(image, str(vertex), (x + int(w / 2) - 5, y + int(h / 2) + 5), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (0, 0, 255), lineType=cv2.LINE_AA)

                    if vertex == 2:
                        _num_of_shapes["line"] += 1
                    elif vertex == 3:
                        _num_of_shapes["triangle"] += 1
                    elif vertex == 4:
                        (x, y, w, h) = cv2.boundingRect(approx)
                        ar = w / float(h)
                        if ar >= 0.5 and ar <= 1.5:
                            _num_of_shapes["rectangle"] += 1
                        else:
                            _num_of_shapes["line"] += 1
                    else:
                        _num_of_shapes["circle"] += 1
            return image, _num_of_shapes["circle"], _num_of_shapes["line"], _num_of_shapes["triangle"], _num_of_shapes["rectangle"]

    def show(self, img, circle, line, triangle, rectangle):
        cv2.circle(img, (150, 40), 20, (0, 0, 255), -1)
        triangle_cnt = np.array([(150, 90), (130, 120), (170, 120)])
        cv2.line(img, (130, 180), (170, 180), (0, 0, 255), 4)
        cv2.rectangle(img, (170, 230), (130, 270), (0, 0, 255), -1)

        cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 255), -1)
        cv2.putText(img, str(circle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(triangle), (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(line), (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(rectangle), (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('image', img)
        cv2.waitKey(1)

    def run(self, image):
        white = self.white_mask(image)
        board = self.shape_mask(white)
        cv2.imshow("board", board)
        cv2.waitKey(0)

        frame, circle, line, triangle, rectangle= self.detection(board, True)
        if frame is None:
            print('none')
            return None, 0, 0, 0, 0
        else:
            self.show(frame, _circle, _line, _triangle, _rectangle)
            print(circle, line, triangle, rectangle)
            return frame, circle, line, triangle, rectangle

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)

    while True:
        ret, img = cap.read()
        mission = ROV([0, 0, 0], [70, 70, 70])
        print("A", img.shape)
        img = cv2.imread("photo/img1.jpg", cv2.IMREAD_COLOR)
        frame, circle, line, triangle, rectangle = mission.run(img)
        if frame is None:
            continue
        else:
            k = cv2.waitKey(1)

            if k == 32:
                _circle = circle
                _line = line
                _triangle = triangle
                _rectangle = rectangle
                mission.show(frame, _circle, _line, _triangle, _rectangle)
            if k == 27:
                print("stop")
                break
            else:
                continue
    cap.release()
    cv2.destroyAllWindows()
