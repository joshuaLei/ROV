import cv2

image = cv2.imread('test.jpg', cv2.IMREAD_COLOR)

img = cv2.resize(image, (900, 480))

h, w, c = img.shape

h3 = int(h/3)
w3 = int(w/3)

grey = img[h3: -h3, w3: -w3, :]
grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)

img[h3: -h3, w3: -w3, 0] = grey
img[0: h3, 0: w3, 1] = 250
img[h3: -h3, 0: w3, 2] = 250
img[-h3:, 0: w3, 0] = 250
img[0: h3, w3: -w3, 0] = 175
img[0: h3, w3: -w3, 1] = 175
img[-h3:, w3: -w3, 1] = 175
img[-h3:, w3: -w3, 2] = 175
img[0: h3, -w3:, 1] = 250
img[h3: -h3, -w3:, 2] = 250
img[-h3:, -w3:, 0] = 250


cv2.imshow("image", img)
cv2.waitKey(0)
