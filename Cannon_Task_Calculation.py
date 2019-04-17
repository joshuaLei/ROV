import os
import cv2 as cv

m = 0

for roots, subfolders, filenames in os.walk("photos"):
    for file in filenames:
        image = file[13:19]
        m = image
        if m > image:
            m = image

print(file)

img = cv.imread("photos/%s" % file)
cv.imshow("img", img)
cv.waitKey(0)
