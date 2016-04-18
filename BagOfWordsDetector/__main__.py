import cv2 as cv

img = cv.imread('img/motorcycle001.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sifter = cv.SURF.detect(gray)
key_points = sifter.detect(gray, None)
cv.drawKeypoints(img, key_points, img)
cv.imshow('window', img)

img = cv.imread('img/motorcycle001.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sifter = cv.FastFeatureDetector_create()
key_points = sifter.detect(gray, None)
cv.drawKeypoints(img, key_points, img)
cv.imshow('window2', img)

cv.waitKey()
