import cv2
import detector

motorcycle_detector = detector.ObjectDetector()
motorcycle_detector.train('./img/neg', './img/pos')

print('\nMagical Predictions:')
test_img = cv2.imread('./img/pos/motorcycle001.jpg')
print(motorcycle_detector.predict(test_img))
test_img = cv2.imread('./img/pos/motorcycle005.jpg')
print(motorcycle_detector.predict(test_img))
test_img = cv2.imread('./img/pos/motorcycle010.jpg')
print(motorcycle_detector.predict(test_img))
test_img = cv2.imread('./img/neg/nature001.jpg')
print(motorcycle_detector.predict(test_img))
test_img = cv2.imread('./img/neg/nature003.jpg')
print(motorcycle_detector.predict(test_img))

print('\nReal Test Images')

print('\nPositive Tests')
test_images_path = detector.directory_filenames('./img/test_pos')
for img_path in test_images_path:
    img = cv2.imread(img_path)
    print(motorcycle_detector.predict(img), ' ', img_path)

print('\nNegative Tests')
test_images_path = detector.directory_filenames('./img/test_neg')
for img_path in test_images_path:
    img = cv2.imread(img_path)
    print(motorcycle_detector.predict(img))
