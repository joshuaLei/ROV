import cv2 as cv
import time
import random
import math
#from video import Video


class ImageTool(object):

    def __init__(self):
        self.video1 = cv.VideoCapture(0)
        #self.video2 = Video(port=4777)
        self.frame = None
        self.cropped = None

        self.pause = False
        self.mode = 0
        self.srcframe = None

        self.ref_scale_rate = 1
        self.ref_pos_start = (0, 0)
        self.ref_pos_end = (0, 0)

        self.tmp_pos_start = []
        self.tmp_pos_end = []

        self.detect_color_from = (0, 0, 0)
        self.detect_color_to = (255, 255, 255)
        self.e1 = 60
        self.e2 = 60
        self.e3 = 60

        self.ref = 0
        self.tmp = 0
        self.ref_real = 7#25#2
        self.tmp_real = 0

        self.ref_val = 6

    def debug(self):
        success, self.frame = self.video1.read()
        self.frame = cv.flip(self.frame, 3)
        return self.frame

    def overlay(self, frame):
        overlay = frame.copy()
        cropped = frame.copy()
        img = frame[120:480, 120:680].copy()
        cv.rectangle(overlay, (120, 120), (680, 480), (0, 0, 255), -1)
        cv.addWeighted(overlay, 0.3, cropped, 1 - 0.3, 0, cropped)
        self.cropped = img
        self.frame = cropped

    def on_mouse_frame(self, event, x, y, flags, param):
        if self.pause:
            if event == cv.EVENT_RBUTTONDOWN:
                self.mode = 1
                self.ref_pos_start = (x, y)
                self.tmp_pos_start = []
                self.tmp_pos_end = []

            if event == cv.EVENT_RBUTTONUP:
                self.mode = 0

            if event == cv.EVENT_LBUTTONDOWN:
                self.mode = 2
                self.tmp_pos_start.append((x, y))

            if event == cv.EVENT_LBUTTONUP:
                self.mode = 0
                self.tmp_pos_end.append((x, y))

            if event == cv.EVENT_MOUSEMOVE:
                if self.mode == 1:
                    image = self.srcframe.copy()
                    self.ref_pos_end = (x, y)
                    cv.line(image, self.ref_pos_start, self.ref_pos_end, (0, 255, 255), 1, cv.LINE_AA)
                    self.frame = image

                if self.mode == 2:
                    image = self.srcframe.copy()
                    cv.line(image, self.ref_pos_start, self.ref_pos_end, (0, 255, 255), 1, cv.LINE_AA)
                    cv.line(image, self.tmp_pos_start[-1], (x, y), (0, 255, 0), 1, cv.LINE_AA)
                    for i in range(len(self.tmp_pos_end)):
                        cv.line(image, self.tmp_pos_start[i], self.tmp_pos_end[i], (0, 255, 0), 1, cv.LINE_AA)
                    self.frame = image
        else:
            if event == cv.EVENT_LBUTTONDBLCLK:
                #temp = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
                h, s, v = self.frame[y, x]
                h = int(h)
                s = int(s)
                v = int(v)
                print('b, g, r',h,s,v)
                self.detect_color_from = (max(h - self.e1, 0), max(s - self.e2, 0), max(v - self.e3, 0))
                self.detect_color_to = (min(h + self.e1, 255), min(s + self.e2, 255), min(v + self.e3, 255))
            if event == cv.EVENT_RBUTTONDBLCLK:
                self.detect_color_from = (0, 0, 0)
                self.detect_color_to = (180, 255, 255)

    def calculation_length(self, Rp_start, Rp_end, Tp_start, Tp_end):
        #get ref point x, y
        xRS = Rp_start[0]
        yRS = Rp_start[1]
        xRE = Rp_end[0]
        yRE = Rp_end[1]
        ref_line = math.sqrt(math.pow((xRS - xRE), 2) + math.pow((yRS - yRE), 2))
        self.ref = ref_line
        print('ref_line:', ref_line)

        #get tmp point x, y
        xTS = Tp_start[-1][0]
        yTS = Tp_start[-1][1]
        xTE = Tp_end[-1][0]
        yTE = Tp_end[-1][1]
        tmp_line = math.sqrt(math.pow((xTS - xTE), 2) + math.pow((yTS - yTE), 2))
        self.tmp = tmp_line
        print('tmp_line', tmp_line)

    def calculation_result(self, ref, tmp):
        print('ref_real', self.ref_real)
        self.tmp_real = ((self.ref_real * tmp)/ref)
        print("result", self.tmp_real)

if __name__ == "__main__":
    tool = ImageTool()
    cv.namedWindow("frame")
    cv.setMouseCallback("frame", tool.on_mouse_frame)

    #video = Video(port=4777)

    i = 0

    while True:
        if not tool.pause:
            #frame = tool.capture()
            frame = tool.debug()
            #if not video.frame_available():
                #continue

            #frame = video.frame()
            tool.srcframe = frame
            frame = cv.resize(frame, (800, 600))
            tool.overlay(frame)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, tool.detect_color_from, tool.detect_color_to)
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(frame, contours, -1, (0, 255, 0), 2)
        else:
            tool.overlay(frame)
            frame = tool.frame

        cv.imshow("frame", frame)

        key = cv.waitKey(1)

        if key == 27:
            break
        if key == 32:
            tool.pause = not tool.pause
            #frame = tool.capture()
            tool.srcframe= frame.copy()
        if key == ord('s'):
            file = "photos/IMG_%s_%d.jpg" % (time.strftime("%Y%m%d_%H%M%S", time.localtime()), random.randint(1, 1000))
            cv.imwrite(file, tool.srcframe)

        if key == ord('c'):
            tool.calculation_length(tool.ref_pos_start, tool.ref_pos_end, tool.tmp_pos_start, tool.tmp_pos_end)
            tool.calculation_result(tool.ref, tool.tmp)
            #print('tmp start:', tool.tmp_pos_start)
            #print('tmp end:', tool.tmp_pos_end)

    #tool.video1.release()
    cv.destroyAllWindows()