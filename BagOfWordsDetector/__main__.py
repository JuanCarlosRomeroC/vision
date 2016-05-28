"""
    Usage: python . <command> (<argument>)

        command: can be one of the four "train", "predict", "predict_all", "predict_dir", "roc"

            train mode trains with default parameters, and using the images in the pos and neg folders,
                and saves the classifier to a file motorcycleDetector.pkl, so that it can be loaded later.
            predict takes the given image in the <argument> parameter and returns True or False if the image is  t
                predicted to be a motorcycle or not.
            predict_all checks all the images in the test_pos and test_neg directories.
            predict_dir takes all the images in the directory specified by the <argument> parameter, and outputs the
                prediction for each one (whether it has a motorcycle or not). It later outputs the percentage of
                images that were predicted to have a motorcycle.
            ROC uses the parameters used to create the ROC curve described in the report.

        img_path: used only if command is predict. Is the target image to predict a motorcycle is inside.
"""
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt

import cv2
import detector


def directory_filenames(dir_name):
    ans = []
    for (dirpath, _, filenames) in os.walk(dir_name):
        ans = [dirpath + '/' + f for f in filenames]
    return ans


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def get_object(filename):
    try:
        with open(filename, 'rb') as input_:
            ans = pickle.load(input_)
        return ans
    except FileNotFoundError:
        print('You must first train the classifier before trying to predict using image files.')
        exit(1)


def detect_all(object_detector, test_neg_dir='./img/test_neg', test_pos_dir='./img/test_pos', to_print=True,
               difficulty=None):
    """
    Uses the detector to identify all the test images. Then calculates (precision, recall)
    :param object_detector: A detector that implements the `predict` method for images
    :param test_neg_dir: A directory with images that are supposed to return False
    :param test_pos_dir: A directory with images that are supposed to return True
    :param difficulty: Only try positive images with the given difficulty (marked by the filename).
     Default - detect all.
    :return: A tuple (precision, recall) describing the results of the detection
    """

    def print_(*args):
        if to_print:
            for arg in args:
                print(arg, end='')
            print()

    print_('\nPositive Tests')
    test_images_path = directory_filenames(test_pos_dir)
    if difficulty is not None:
        if difficulty not in [1, 2, 3]:
            raise Exception('No such difficulty level.')
        test_images_path = [p for p in test_images_path if re.search('dif' + str(difficulty), p) is not None]
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
    test_images_path = directory_filenames(test_neg_dir)
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
                    train_pos_dir='./img/pos', to_print=True):
    if to_train:
        motorcycle_detector = detector.ObjectDetector(to_print=to_print)
        motorcycle_detector.train(train_neg_dir, train_pos_dir)
        if to_save:
            save_object(motorcycle_detector, filename)
        return motorcycle_detector
    else:
        return get_object(filename)


def create_roc():
    points_all = []
    points_1 = []
    points_2 = []
    points_3 = []
    for features_num in range(6, 30, 3):
        for k_neighbors in range(3, 15, 3):
            detector.ObjectDetector.PARAMS = {
                'FEATURES_NUM': features_num,
                'N_NEIGHBORS_DEFAULT': k_neighbors
            }
            det = create_detector(to_train=True, to_save=False, to_print=False)
            precision, recall = detect_all(det, to_print=False)
            points_all.append((precision, recall))
            points_1.append(
                detect_all(det, to_print=False, difficulty=1))
            points_2.append(
                detect_all(det, to_print=False, difficulty=2))
            points_3.append(
                detect_all(det, to_print=False, difficulty=3))
            print('features_num, k_neighbors are: {}, precision, recall are: {}'.format((features_num, k_neighbors),
                                                                                        (precision, recall)))
    x, y = [p[0] for p in points_all], [p[1] for p in points_all]
    plt.scatter(x, y, c='black')
    x, y = [p[0] for p in points_1], [p[1] for p in points_1]
    plt.scatter(x, y, c='red')
    x, y = [p[0] for p in points_2], [p[1] for p in points_2]
    plt.scatter(x, y, c='green')
    x, y = [p[0] for p in points_3], [p[1] for p in points_3]
    plt.scatter(x, y, c='blue')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Wrong number of arguments.')
        print(__doc__)
        exit(1)
    mode = sys.argv[1]
    if mode == 'train':
        create_detector(to_train=True, to_save=True)
    elif mode == 'predict':
        det = create_detector(to_train=False, to_save=False)
        if len(sys.argv) < 3:
            print('The predict command required the parameter <IMG>.')
            exit(1)
        img = cv2.imread(sys.argv[2])
        if img is None:
            print('Invalid image file for predict command.')
            exit(1)
        print('Is this a motorcycle? The prediction is {}'.format(det.predict(img)))
    elif mode == 'predict_all':
        det = create_detector(to_train=False, to_save=False)
        precision, recall = detect_all(det)
        print()
        print('The precision value is {} and the recall is {}'.format(precision, recall))
    elif mode == 'ROC':
        create_roc()
    elif mode == 'predict_dir':
        if len(sys.argv) < 3:
            print('The predict_dir command required the parameter <DIR>.')
            exit(1)
        print('Predictions for all images in the directory {}:'.format(sys.argv[2]))
        det = create_detector(to_train=False, to_save=False)
        motors = 0
        images = directory_filenames(sys.argv[2])
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                print('The file at {} is not a valid image file. Aborting.'.format(img_path))
                exit(1)
            has_motorcycle = det.predict(img)
            if has_motorcycle:
                motors += 1
            print('{path}: {prediction}'.format(path=img_path, prediction='yes' if has_motorcycle else 'no'))
        print('\nThe percentage of motorcycles is {}'.format(motors / len(images)))
    else:
        print('Illegal arguments: Unknown mode.')
        print(__doc__)
        exit(1)
