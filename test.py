import cv2
import numpy as np

def nothing(x):
    pass

def mask(frame, lower, upper):
    bgr = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(bgr, lower, upper)
    return mask

cv2.namedWindow('image')
# create trackbars for color change
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)


while(1):
    img = cv2.imread("img/phot.jpg")
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()