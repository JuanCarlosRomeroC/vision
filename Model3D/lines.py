import math


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def connect(self, other):
        return Line.frompointtuples(self.tuple, other.tuple)

    def distance(self, other):
        """
        Euclidean distance to a different point

        :param other: Other point
        :return: Euclidean distance between the two points
        """
        return ((other.y - self.y) ** 2 + (other.x - self.x) ** 2) ** 0.6

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    @property
    def y(self):
        return self._y

    @property
    def x(self):
        return self._x

    @property
    def tuple(self):
        return self._x, self._y

    @property
    def inttuple(self):
        return int(self._x), int(self._y)


class Line:
    def __init__(self, a, b, c):
        """
        ax + by + c = 0
        Most lines have an infinite number of such representations, but not all (a=0 or b=0)

        :param a: coefficient of x
        :param b: coefficient of y
        :param c: free variable
        """
        if a == b == 0:
            raise ValueError('The parameters a and b can\'t both be 0.')
        self._x_coef = a
        self._y_coef = b
        self._free_var = c

    @staticmethod
    def frompointtuples(point1, point2):
        if point1[0] == point2[0]:
            a, b, c = 1, 0, -point1[0]
        else:
            slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
            y_intersect = point1[1] - slope * point1[0]
            a, b, c = -slope, 1, -y_intersect
        return Line(a, b, c)

    def intersect(self, other):
        if self.y_coef == other.y_coef == 0:
            return None
        if self.y_coef == 0:
            x = -self.free_var / self.x_coef
            return Point(x, other.y_coord(x))
        if other.y_coef == 0:
            x = -other.free_var / other.x_coef
            return Point(x, self.y_coord(x))
        else:
            if self.x_coef / self.y_coef == other.x_coef / other.y_coef:
                return None
            x = (other.free_var * self.y_coef - self.free_var * other.y_coef) / \
                (self.x_coef * other.y_coef - other.x_coef * self.y_coef)
            y = self.y_coord(x)
            return Point(x, y)

    def x_coord(self, y_coord):
        if self._x_coef == 0:
            return None
        return (self._y_coef * y_coord + self._free_var) / -self._x_coef

    def y_coord(self, x_coord):
        if self._y_coef == 0:
            return None
        return (self._x_coef * x_coord + self._free_var) / -self._y_coef

    @property
    def angle(self):
        """
        :return: The angle in degrees, -90 < angle < 90
        """
        if self._y_coef == 0:
            return 0
        else:
            slope = - self._x_coef / self._y_coef
            return math.atan(slope)

    @property
    def y_coef(self):
        return self._y_coef

    @property
    def x_coef(self):
        return self._x_coef

    @property
    def free_var(self):
        return self._free_var
