import os

from sklearn.svm import SVC
import numpy as np

import cv2


def directory_filenames(dir_name):
    ans = []
    for (dirpath, _, filenames) in os.walk(dir_name):
        ans = [dirpath + '/' + f for f in filenames]
    return ans


class ObjectDetector:
    PARAMS = {
        'FEATURES_NUM': 20
    }

    def __init__(self):
        self._k_features = None
        self._classifier = None

    @staticmethod
    def _closest_point(goal, points):
        points = np.asarray(points)
        dist_2 = np.sum((points - goal) ** 2, axis=1)
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
        return [(p.angle, p.pt[0], p.pt[1], p.size * 10) for p in key_points]

    def train_classifier(self, negative_vectors, positive_vectors):
        all_vectors = []
        all_vectors.extend(negative_vectors)
        all_vectors.extend(positive_vectors)
        # create an array of expected results that looks like this: [-1, -1, -1, -1 ..., 1, 1, 1, 1]
        expected_results = np.where(np.arange(start=1, stop=len(all_vectors) + 1) <= len(negative_vectors), -1, 1)

        self._classifier = SVC().fit(all_vectors, expected_results)

    def set_k_features(self, positive_keypoints):
        samples = np.array(positive_keypoints, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_MAX_ITER, 20, 1.)
        _, _, self._k_features = cv2.kmeans(samples, ObjectDetector.PARAMS['FEATURES_NUM'], None, criteria, 10,
                                            cv2.KMEANS_RANDOM_CENTERS)

    def feature_vector(self, img):
        if self._k_features is None:
            raise Exception('Cannot calculate feature_vector before setting the k features (using positive examples)')
        features = self._k_features
        img_features = np.zeros(len(features))
        for keypoint in ObjectDetector._img_keypoint_vectors(img):
            img_features[ObjectDetector._closest_point(keypoint, features)] += 1
        return img_features

    def train(self, neg_input_dir, pos_input_dir):
        positive_images_paths = directory_filenames(pos_input_dir)
        negative_images_paths = directory_filenames(neg_input_dir)
        positive_keypoints = []

        # Gather all keypoints of positive images
        i = 1
        for img_path in positive_images_paths:
            img = cv2.imread(img_path)
            positive_keypoints.extend(ObjectDetector._img_keypoint_vectors(img))

            print('Calculating features... {:.0f}%'.format(100. * i / len(positive_images_paths)))
            i += 1

        # Cluster all the keypoints to k features
        self.set_k_features(positive_keypoints)

        # Gather positive feature vectors and negative feature vectors
        positive_feature_vectors = []
        negative_feature_vectors = []
        i = 1
        for img_path in positive_images_paths:
            img = cv2.imread(img_path)
            positive_feature_vectors.append(self.feature_vector(img))

            print('Calculating feature vectors for positive points... {:.0f}%'.format(
                100. * i / len(positive_images_paths)))
            i += 1
        i = 1
        for img_path in negative_images_paths:
            img = cv2.imread(img_path)
            negative_feature_vectors.append(self.feature_vector(img))

            print('Calculating feature vectors for negative points... {:.0f}%'.format(
                100. * i / len(negative_images_paths)))
            i += 1

        # Train classifier
        self.train_classifier(negative_feature_vectors, positive_feature_vectors)

    def predict(self, img):
        if self._classifier is None:
            raise Exception('Object detector cannot predict before being trained')
        else:
            unknown = np.stack((self.feature_vector(img),))
            return self._classifier.predict(unknown)[0]


