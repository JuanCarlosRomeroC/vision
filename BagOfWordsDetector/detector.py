import os

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import cv2


# ------ Constants -------

HEIGHT, WIDTH = 0, 1
KEYPOINT_X, KEYPOINT_Y = 0, 1
NOT_OBJECT, OBJECT = -1, 1


# ---------- Helpers -----------

def directory_filenames(dir_name):
    ans = []
    for (dirpath, _, filenames) in os.walk(dir_name):
        ans = [dirpath + '/' + f for f in filenames]
    return ans


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def display_points(img, points, color=(0, 255, 0), size=5, window_name='Points Display'):
    img_copy = np.copy(img)
    for point in points:
        print((int(point.x), int(point.y)))
        cv2.circle(img_copy, (int(point.x), int(point.y)), radius=size, color=color, thickness=3)
        cv2.imshow(window_name, img_copy)
        cv2.waitKey(5)
    cv2.imshow(window_name, img_copy)


def show_siftpoints(img):
    sifter = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key_points = sifter.detect(gray, None)
    img_points = np.copy(img)
    cv2.drawKeypoints(img, key_points, img_points)
    cv2.imshow('Siftpoints', img_points)


# -------- Detector ---------

positive_keypoints_cache = None


class ObjectDetector:
    PARAMS = {
        'FEATURES_NUM': 12,
        'N_NEIGHBORS_DEFAULT': 9
    }

    def __init__(self, bias=(1, 1), to_print=True):
        """
        Create an Object Detector that can be trained with images and then predict for a new image.
        :param bias: If the objects are twice more popular than the non-objects, for example, one can give a bias.
         Bias is a tuple, with negative bias then positive bias. So, (2, 1) means objects are twice more popular.
        :param to_print: If the object detector should print debug messages as it trains
        """
        self._k_features = None
        self._classifier = None
        self.to_print = to_print
        self.bias = bias

    @staticmethod
    def _closest_point(goal, points):
        points = np.asarray(points)
        dist_2 = np.sum(np.square((points - goal)), axis=1)
        return np.argmin(dist_2)

    @staticmethod
    def _img_keypoint_vectors(img):
        """
        :param img: an image (2d matrix)
        :return: list of vectors, each one representing a keypoint in the image (SIFT KeyPoint)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sifter = cv2.xfeatures2d.SIFT_create()
        key_points = sifter.detect(gray, None)
        width = float(gray.shape[WIDTH])
        height = float(gray.shape[HEIGHT])

        estimated_size_factor = (width + height) / 2
        return [(p.pt[KEYPOINT_Y] / height, p.pt[KEYPOINT_X] / width) for p in key_points]

    def print(self, *args):
        if self.to_print:
            for arg in args:
                print(arg, end='')
            print()

    def train_classifier(self, negative_vectors, positive_vectors):
        # enlarge the two vectors by bias
        negative_vectors = negative_vectors * self.bias[0]
        positive_vectors = positive_vectors * self.bias[1]

        all_vectors = []
        all_vectors.extend(negative_vectors)
        all_vectors.extend(positive_vectors)
        # create an array of expected results that looks like this: [-1, -1, -1, -1 ..., 1, 1, 1, 1]
        expected_results = np.where(np.arange(start=1, stop=len(all_vectors) + 1) <= len(negative_vectors), NOT_OBJECT,
                                    OBJECT)

        k_neighbors = ObjectDetector.PARAMS['N_NEIGHBORS_DEFAULT'] * (self.bias[0] + self.bias[1]) // (1 + 1)
        self._classifier = KNeighborsClassifier(k_neighbors).fit(all_vectors, expected_results)
        # self._classifier = RandomForestClassifier().fit(all_vectors, expected_results)

    def set_k_features(self, positive_keypoints):
        samples = np.array(positive_keypoints, dtype=np.float32)

        iterations = 20
        repetitions = 10
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.)
        _, _, self._k_features = cv2.kmeans(samples, ObjectDetector.PARAMS['FEATURES_NUM'], None, criteria, repetitions,
                                            cv2.KMEANS_RANDOM_CENTERS)

    def display_k_features(self, img):
        height = float(img.shape[HEIGHT])
        width = float(img.shape[WIDTH])
        self.print('K features:', self._k_features)
        points = [Point(p[WIDTH] * width, p[HEIGHT] * height) for p in self._k_features]
        display_points(img, points)

    def feature_vector(self, img):
        if self._k_features is None:
            raise Exception('Cannot calculate feature_vector before setting the k features (using positive examples)')
        features = self._k_features
        img_features = np.zeros(len(features))
        for keypoint in ObjectDetector._img_keypoint_vectors(img):
            img_features[ObjectDetector._closest_point(keypoint, features)] += 1
        return img_features

    def train(self, neg_input_dir, pos_input_dir):
        global positive_keypoints_cache

        positive_images_paths = directory_filenames(pos_input_dir)
        negative_images_paths = directory_filenames(neg_input_dir)

        # Gather all keypoints of positive images
        positive_keypoints = positive_keypoints_cache
        if positive_keypoints is None:
            positive_keypoints = []
            i = 1
            for img_path in positive_images_paths:
                img = cv2.imread(img_path)
                positive_keypoints.extend(ObjectDetector._img_keypoint_vectors(img))

                self.print('Calculating features... {:.0f}%'.format(100. * i / len(positive_images_paths)))
                i += 1
            positive_keypoints_cache = positive_keypoints

        # Cluster all the keypoints to k features
        self.set_k_features(positive_keypoints)

        # Gather positive feature vectors and negative feature vectors
        positive_feature_vectors = []
        negative_feature_vectors = []
        i = 1
        for img_path in positive_images_paths:
            img = cv2.imread(img_path)
            positive_feature_vectors.append(self.feature_vector(img))

            self.print('Calculating feature vectors for positive points... {:.0f}%'.format(
                100. * i / len(positive_images_paths)))
            i += 1
        i = 1
        for img_path in negative_images_paths:
            img = cv2.imread(img_path)
            negative_feature_vectors.append(self.feature_vector(img))

            self.print('Calculating feature vectors for negative points... {:.0f}%'.format(
                100. * i / len(negative_images_paths)))
            i += 1

        # Train classifier
        self.train_classifier(negative_feature_vectors, positive_feature_vectors)

    def predict(self, img):
        if self._classifier is None:
            raise Exception('Object detector cannot predict before being trained')
        else:
            unknown = np.stack((self.feature_vector(img),))
            if self._classifier.predict(unknown)[0] == OBJECT:
                return True
            img = np.fliplr(img)
            unknown = np.stack((self.feature_vector(img),))
            if self._classifier.predict(unknown)[0] == OBJECT:
                return True
            return False
