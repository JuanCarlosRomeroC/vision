import math

import utils


class Model3D:
    def __init__(self):
        self._horizon = None
        self._vertical_top = None
        self._known_size = None

    def set_horizon(self, parallel_one, parallel_two):
        """
        Calculates the horizon in the picture

        :param parallel_one: A list of lines that in 3d are parallel to one another and parallel to the ground plane
        :param parallel_two: A list of lines that in 3d are parallel to one another and parallel ot the ground plane
         but not to lines from the list parallel_one
        """
        horizon_point1 = utils.Averager()
        for l1, l2 in utils.choose_two(parallel_one):
            horizon_point1.add_element(l1.intersect(l2))
        horizon_point1 = horizon_point1.outcome

        horizon_point2 = utils.Averager()
        for l1, l2 in utils.choose_two(parallel_two):
            horizon_point2.add_element(l1.intersect(l2))
        horizon_point2 = horizon_point2.outcome

        self._horizon = horizon_point1.connect(horizon_point2)

    def set_vertical_top(self, parallel_vertical):
        """
        Calculates the vertical top in the picture (could be infinity)

        :param parallel_vertical: A list of lines that in 3d are vertical (and parallel to one another)
        """

        def almost_parallel(line1, line2):
            difference = math.fabs(line1.angle - line2.angle)
            threshold = 3
            return difference < threshold or difference > 180 - threshold

        almost_verticle_count = 0
        for line1, line2 in utils.choose_two(parallel_vertical):
            if almost_parallel(line1, line2):
                almost_verticle_count += 1
        if almost_verticle_count >= len(parallel_vertical) / 2:
            self._vertical_top = None
            return

        self._vertical_top = utils.Averager()
        for line1, line2 in utils.choose_two(parallel_vertical):
            self._vertical_top.add_element(line1.intersect(line2))
        self._vertical_top = self._vertical_top.outcome

    def set_known_size(self, point1, point2, size):
        """
        Feed the model with a known size in the image

        :param point1: Bottom of the object
        :param point2: Top of the object
        :param size: The size of the object
        """
        self._known_size = (point1, point2, size)

    def calculate_distance(self, point1, point2):
        """
        Calculates 3D distance between two points in the image

        :param point1: Start of object (bottom)
        :param point2: End of object (top)
        :return: Size of object in 3D
        """
        if self._horizon is None or self._known_size is None:
            raise Exception('Not enough data to calculate distance.')
        known_bottom, known_top, known_size = self._known_size
        known_line = known_bottom.connect(known_top)

        feet_line = point1.connect(known_bottom)
        vanishing = feet_line.intersect(self._horizon)
        head_line = vanishing.connect(point2)

        height_point = head_line.intersect(known_line)
        if self._vertical_top is None:
            return known_size * (known_bottom.distance(height_point) / known_bottom.distance(known_top))
        else:
            camera_tilt_fix = self._vertical_top.distance(known_top) / self._vertical_top.distance(height_point)
            return known_size * (
                known_bottom.distance(height_point) / known_bottom.distance(known_top)) * camera_tilt_fix
