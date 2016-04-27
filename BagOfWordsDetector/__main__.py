import pickle

import numpy as np
import matplotlib.pyplot as plt

import cv2
import detector


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def get_object(filename):
    with open(filename, 'rb') as input_:
        ans = pickle.load(input_)
    return ans


def detect_all(object_detector, test_neg_dir='./img/test_neg', test_pos_dir='./img/test_pos', to_print=True):
    """
    Uses the detector to identify all the test images. Then calculates (precision, recall)
    :param object_detector: A detector that implements the `predict` method for images
    :param test_neg_dir: A directory with images that are supposed to return False
    :param test_pos_dir: A directory with images that are supposed to return True
    :return: A tuple (precision, recall) describing the results of the detection
    """

    def print_(*args):
        if to_print:
            for arg in args:
                print(arg, end='')
            print()

    print_('\nPositive Tests')
    test_images_path = detector.directory_filenames(test_pos_dir)
    n_pos_samples = len(test_images_path)
    true_positive_count = 0
    for img_path in test_images_path:
        img = cv2.imread(img_path)
        prediction = object_detector.predict(img)
        if prediction:
            true_positive_count += 1
        if to_print:
            print(prediction, ' ', img_path)

    print_('\nNegative Tests')
    test_images_path = detector.directory_filenames(test_neg_dir)
    false_positive_count = 0
    for img_path in test_images_path:
        img = cv2.imread(img_path)
        prediction = object_detector.predict(img)
        if prediction:
            false_positive_count += 1
        print_(prediction, ' ', img_path)

    precision = float(true_positive_count) / (true_positive_count + false_positive_count)
    recall = float(true_positive_count) / n_pos_samples
    return precision, recall


def create_detector(filename='motorcycleDetector.pkl', to_train=False, to_save=False, train_neg_dir='./img/neg',
                    train_pos_dir='./img/pos'):
    if to_train:
        motorcycle_detector = detector.ObjectDetector()
        motorcycle_detector.train(train_neg_dir, train_pos_dir)
        if to_save:
            save_object(motorcycle_detector, filename)
        return motorcycle_detector
    else:
        return get_object(filename)


def create_roc():
    points = []
    for k in range(2, 7):
        detector.ObjectDetector.PARAMS['FEATURES_NUM'] = k
        motorcycle_detector = create_detector(to_train=True, to_save=False)
        precision, recall = detect_all(motorcycle_detector)
        print('Precision, Recall:', precision, recall)
        points.append((precision, recall))
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    plt.plot(x, y)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision/Recall as Number of Features Change')
    plt.grid(True)
    plt.show()


detect_all(create_detector(to_train=True, to_save=True), to_print=True)
