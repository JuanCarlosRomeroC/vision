"""
    This module allows the user to interactively build a 3D-model of an input image, and then calculate the height of
    different objects in the image based on one known height.

    Usage:
        python run.py <img_path>

    After the image has loaded, you may start marking lines in the image. Four different marking types are needed:
    * '1' - Lines that in reality are parallel to one another and parallel to the ground.
            These lines should not be parallel in the image.
    * '2' - Lines that in reality are parallel to one another and parallel to the ground, and not parallel to '1'
            These lines should not be parallel in the image.
    * 'v' - Lines that are vertical in reality
    * 'k' - A known-height object. This is an object in the image with a known height.
            When you mark this object, ALWAYS mark the BOTTOM first and then the top.

    By pressing on the keyboard on one of the above characters, you will enter the relevant marking mode. After you have
    chosen a mode, you can draw lines by clicking on where the line should start, and then clicking on where the line
    should end. At any time, you may press the character 'z', which will allow you to zoom. After you press 'z', your
    pen will turn gray, and you will be able to mark a rectangle to zoom to on the image. To zoom back out, press the
    character 'z' again.

    After you have marked all four types, you will be able to measure heights in the image. Press the character 'u',
    that stands for "unknown size" (as opposed to the known size 'k'). Then mark the object you would like to measure.
    Notice - when you mark this object, ALWAYS mark the BOTTOM first and then the top.

    After measuring everything you every wished to measure, press ESC to exit the application.
"""

import sys
from enum import Enum, unique

import cv2
import model
import pymsgbox
import scipy.misc

from Model3D import lines

ESC_KEY = 27
i = 0


class ModelCreator:
    def __init__(self, img):
        self._img = img
        self._model = model.Model3D()
        self._state = ModelCreator.State.INIT

        self._horizontal_1 = []  # a list of (point, point)
        self._horizontal_2 = []  # a list of (point, point)
        self._vertical = []  # a list of (point, point)
        self._known = None  # (point1, point2, length)
        self._unknown = None  # (point1, point2)

        self._zoom_scale_factor = None
        self._zoom_start = None  # (imgpoint_x, imgpoint_y)
        self._zoom_end = None  # (imgpoint_x, imgpoint_y)
        self._state_before_zoom = None

        self._cur_mouse_image_pos = (0, 0)  # drawing_helper
        self._current_line_beg = None  # drawing helper
        self._current_rect_beg = None  # drawing helper

        self._window_name = 'Model3D'  # constant

    def run(self):
        cv2.imshow(self._window_name, self._img)
        cv2.setMouseCallback(self._window_name, self.mouse_callback())
        self._state = ModelCreator.State.DRAWING_LINE_HORIZONTAL_1
        while True:
            self.draw_frame()
            k = cv2.waitKey(10)
            if k == ESC_KEY:
                break
            elif k == ord('1'):
                print('Character pressed: 1. Please draw lines parallel to the ground.')
                self._state = ModelCreator.State.DRAWING_LINE_HORIZONTAL_1
            elif k == ord('2'):
                print('Character pressed: 2. Please draw lines differently parallel to the ground.')
                self._state = ModelCreator.State.DRAWING_LINE_HORIZONTAL_2
            elif k == ord('v'):
                print('Character pressed: v. Please draw vertical lines.')
                self._state = ModelCreator.State.DRAWING_LINE_VERTICAL
            elif k == ord('k'):
                print('Character pressed: k. Please mark the object that has a known height. First, mark its bottom,' +
                      ' and then mark its top.')
                self._state = ModelCreator.State.DRAWING_LINE_KNOWNSIZE
            elif k == ord('u'):
                print('Character pressed: u. Please mark the object to measure. First, mark its bottom,' +
                      ' and then mark its top.')
                self._state = ModelCreator.State.DRAWING_LINE_UNKNOWN
            elif k == ord('z'):
                print('Character pressed: z. You may choose a rectangle to zoom into')
                if self._zoom_scale_factor is None:
                    self._state_before_zoom = self._state
                    self._state = ModelCreator.State.DRAWING_RECT_ZOOM
                else:
                    self.zoom_out()

    def draw_frame(self):
        frame = self._img.copy()
        for line in self._horizontal_1:
            cv2.line(frame, line[0], line[1], ModelCreator.colors[ModelCreator.State.DRAWING_LINE_HORIZONTAL_1], 1)
        for line in self._horizontal_2:
            cv2.line(frame, line[0], line[1], ModelCreator.colors[ModelCreator.State.DRAWING_LINE_HORIZONTAL_2], 1)
        for line in self._vertical:
            cv2.line(frame, line[0], line[1], ModelCreator.colors[ModelCreator.State.DRAWING_LINE_VERTICAL], 1)
        if self._known is not None:
            line = self._known
            cv2.line(frame, line[0], line[1], ModelCreator.colors[ModelCreator.State.DRAWING_LINE_KNOWNSIZE], 1)
        if self._unknown is not None:
            line = self._unknown
            cv2.line(frame, line[0], line[1], ModelCreator.colors[ModelCreator.State.DRAWING_LINE_UNKNOWN], 1)
        if self._current_line_beg:
            line = (self._current_line_beg, self._cur_mouse_image_pos)
            cv2.line(frame, line[0], line[1], ModelCreator.colors[self._state], 1)
        if self._current_rect_beg:
            rect = (self._current_rect_beg, self._cur_mouse_image_pos)
            cv2.rectangle(frame, rect[0], rect[1], ModelCreator.colors[ModelCreator.State.DRAWING_RECT_ZOOM], 2)

        if self._zoom_start is not None:
            frame = frame[self._zoom_start[1]:self._zoom_end[1], self._zoom_start[0]:self._zoom_end[0]]
            if self._zoom_scale_factor != 1:
                frame = scipy.misc.imresize(frame, size=self._zoom_scale_factor)
        cv2.imshow(self._window_name, frame)

    def _img_clicked(self, point):
        if self._state.drawing_line():
            if self._current_line_beg is None:
                self._current_line_beg = point
            elif self._current_line_beg != point:
                newline = (self._current_line_beg, point)
                self._current_line_beg = None
                if self._state == ModelCreator.State.DRAWING_LINE_HORIZONTAL_1:
                    self._horizontal_1.append(newline)
                elif self._state == ModelCreator.State.DRAWING_LINE_HORIZONTAL_2:
                    self._horizontal_2.append(newline)
                elif self._state == ModelCreator.State.DRAWING_LINE_VERTICAL:
                    self._vertical.append(newline)
                elif self._state == ModelCreator.State.DRAWING_LINE_KNOWNSIZE:
                    size = pymsgbox.prompt(text='What is the length (cm) of the marked object?', title='Length Input')
                    self._known = (newline[0], newline[1], float(size))
                elif self._state == ModelCreator.State.DRAWING_LINE_UNKNOWN:
                    self._unknown = newline
                    pymsgbox.alert(text='The length is {:.4f}cm.'.format(self.calculate_length()), title='Answer')
        if self._state == ModelCreator.State.DRAWING_RECT_ZOOM:
            if self._current_rect_beg is None:
                self._current_rect_beg = point
            else:
                self.zoom_in(self._current_rect_beg, point)
                self._current_rect_beg = None
                self._state = self._state_before_zoom

    def zoom_in(self, rect_beg, rect_end):
        min_x, max_x = sorted([rect_beg[0], rect_end[0]])
        min_y, max_y = sorted([rect_beg[1], rect_end[1]])
        self._zoom_start = (min_x, min_y)
        self._zoom_end = (max_x, max_y)
        width = max_x - min_x
        height = max_y - min_y
        orig_width = self._img.shape[1]
        orig_height = self._img.shape[0]
        self._zoom_scale_factor = min(orig_width / width, orig_height / height)

    def zoom_out(self):
        self._zoom_scale_factor = None
        self._zoom_start = None
        self._zoom_end = None

    def calculate_length(self):
        horizontal_lines1 = [lines.Line.frompointtuples(l[0], l[1]) for l in self._horizontal_1]
        horizontal_lines2 = [lines.Line.frompointtuples(l[0], l[1]) for l in self._horizontal_2]
        vertical_lines = [lines.Line.frompointtuples(l[0], l[1]) for l in self._vertical]
        self._model.set_horizon(horizontal_lines1, horizontal_lines2)
        self._model.set_vertical_top(vertical_lines)
        self._model.set_known_size(lines.Point(*(self._known[0])), lines.Point(*(self._known[1])), self._known[2])
        return self._model.calculate_distance(lines.Point(*(self._unknown[0])), lines.Point(*(self._unknown[1])))

    def mouse_callback(self):
        def callback(event, x, y, *_):
            if self._zoom_start is not None:
                scale = self._zoom_scale_factor
                start = self._zoom_start
                self._cur_mouse_image_pos = (int(x / scale + start[0]), int(y / scale + start[1]))
            else:
                self._cur_mouse_image_pos = (x, y)

            if event == cv2.EVENT_LBUTTONDOWN:
                self._img_clicked(self._cur_mouse_image_pos)

        return callback

    @unique
    class State(Enum):
        INIT = 0
        DRAWING_LINE_HORIZONTAL_1 = 1
        DRAWING_LINE_HORIZONTAL_2 = 2
        DRAWING_LINE_VERTICAL = 3
        DRAWING_LINE_KNOWNSIZE = 4
        DRAWING_LINE_UNKNOWN = 5
        DRAWING_RECT_ZOOM = 6

        def drawing_line(self):
            return 1 <= self.value <= 5

    colors = {
        State.DRAWING_LINE_HORIZONTAL_1: (255, 255, 0),
        State.DRAWING_LINE_HORIZONTAL_2: (255, 0, 0),
        State.DRAWING_LINE_VERTICAL: (0, 0, 255),
        State.DRAWING_LINE_KNOWNSIZE: (0, 255, 0),
        State.DRAWING_LINE_UNKNOWN: (255, 255, 255),
        State.DRAWING_RECT_ZOOM: (200, 200, 200)
    }


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of parameters.')
        print(__doc__)
        exit(1)
    img_path = sys.argv[1]
    image = cv2.imread(img_path)
    ModelCreator(image).run()
