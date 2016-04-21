import cv2
import BagOfWordsDetector.detector as detector

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
