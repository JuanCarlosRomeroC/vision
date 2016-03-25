import numpy
import cv2


def test(full_path):
    image = cv2.imread(full_path)
    cv2.imshow('test', image)

RED = numpy.array([255, 0, 0], dtype=numpy.uint8)
BLUE = numpy.array([0, 0, 255], dtype=numpy.uint8)
GREEN = numpy.array([0, 255, 0], dtype=numpy.uint8)
BLACK = numpy.array([0, 0, 0], dtype=numpy.uint8)
WHITE = numpy.array([255, 255, 255], dtype=numpy.uint8)

BG = {'color': BLACK, 'val': numpy.uint8(0)}
FG = {'color': WHITE, 'val': numpy.uint8(1)}
PR_BG = {'color': RED, 'val': numpy.uint8(2)}
PR_FG = {'color': GREEN, 'val': numpy.uint8(3)}
UNKNOWN = {'val': numpy.uint8(4)}

CLASSES = [BG, FG, PR_BG, PR_FG]


ESCAPE = 27


class ImageSegmentation:
    def __init__(self, file_name):
        self.input_window_name = 'input'
        self.output_window_name = 'output'

        self.original_img = cv2.imread(file_name)
        self.input_img = self.original_img.copy()
        self.output_img = numpy.zeros(self.input_img.shape, numpy.uint8)

        #  Rectangle
        self.rect = (0, 0, 0, 0)
        self.rect_drawn = False
        self.rect_ready = False
        self.rect_start_x = 0
        self.rect_start_y = 0

        #  Classes
        self.line_class = 0
        self.line_drawn = False
        self.input_mask = numpy.zeros(self.original_img.shape[:2], numpy.uint8)
        self.input_mask += UNKNOWN['val']
        self.classes_matrices = []
        for clazz in CLASSES:
            class_matrix = numpy.ndarray(self.original_img.shape, dtype=numpy.uint8)
            for i in range(class_matrix.shape[0]):
                for j in range(class_matrix.shape[1]):
                    class_matrix[i][j] = clazz['color']
            self.classes_matrices.append(class_matrix)

        # Mask
        self.mask = numpy.zeros(self.original_img.shape[:2], numpy.uint8)
        self.mask += PR_BG['val']

        cv2.namedWindow(self.input_window_name)
        cv2.namedWindow(self.output_window_name)
        cv2.setMouseCallback(self.input_window_name, self.create_mouse_listener())

    def run(self):
        while True:
            self.update_windows()
            key = cv2.waitKey()

            if key == ESCAPE:  # esc  exit
                break
            elif key == ord('0'):
                self.line_class = 0
            elif key == ord('1'):
                self.line_class = 1
            elif key == ord('2'):
                self.line_class = 2
            elif key == ord('3'):
                self.line_class = 3
            print('choose ', (key - ord('0')))
        cv2.destroyAllWindows()

    def create_mouse_listener(self):
        def mouse_listener(event, x, y, flags, param):

            thickness = 3
            rect_color = [255, 0, 0]
            rect_bold = 2

            # Draw Rectangle
            if event == cv2.EVENT_RBUTTONDOWN:
                self.rect_drawn = True
                self.rect_start_x, self.rect_start_y = x, y
                self.input_img = self.original_img.copy()
                start_point = (self.rect[0], self.rect[1])
                end_point = (self.rect[2], self.rect[3])
                cv2.rectangle(self.input_img, start_point, end_point, rect_color, rect_bold)
                self.update_windows()

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.rect_drawn:
                    self.rect = (min(self.rect_start_x, x), min(self.rect_start_y, y), max(self.rect_start_x, x),
                                 max(self.rect_start_y, y))
                    self.input_img = self.original_img.copy()
                    start_point = (self.rect[0], self.rect[1])
                    end_point = (self.rect[2], self.rect[3])
                    cv2.rectangle(self.input_img, start_point, end_point, rect_color, rect_bold)
                    self.update_windows()

            if event == cv2.EVENT_RBUTTONUP:
                if self.rect_drawn:
                    self.rect_drawn = False
                    self.rect_ready = True

                    self.rect = (min(self.rect_start_x, x), min(self.rect_start_y, y), max(self.rect_start_x, x),
                                 max(self.rect_start_y, y))
                    self.input_img = self.original_img.copy()
                    start_point = (self.rect[0], self.rect[1])
                    end_point = (self.rect[2], self.rect[3])
                    cv2.rectangle(self.input_img, start_point, end_point, rect_color, rect_bold)
                    self.update_windows()

            elif event == cv2.EVENT_LBUTTONDOWN:
                if not self.rect_ready:
                    print("first draw rectangle \n")
                else:
                    self.line_drawn = True
                    clazz = CLASSES[self.line_class]
                    cv2.circle(self.input_mask, (x, y), thickness, int(clazz['val']), -1)
                    cv2.circle(self.input_img, (x, y), thickness, clazz['color'].tolist(), -1)
                    self.update_windows()

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.line_drawn:
                    clazz = CLASSES[self.line_class]
                    cv2.circle(self.input_mask, (x, y), thickness, int(clazz['val']), -1)
                    cv2.circle(self.input_img, (x, y), thickness, clazz['color'].tolist(), -1)
                    self.update_windows()

            elif event == cv2.EVENT_LBUTTONUP:
                if self.line_drawn:
                    self.line_drawn = False
                    clazz = CLASSES[self.line_class]
                    cv2.circle(self.input_mask, (x, y), thickness, int(clazz['val']), -1)
                    cv2.circle(self.input_img, (x, y), thickness, clazz['color'].tolist(), -1)
                    self.update_windows()

        return mouse_listener

    def update_windows(self):
        # extended_input_mask = numpy.stack((self.input_mask, self.input_mask, self.input_mask), axis=2)
        # for clazz in CLASSES:
        #     self.input_img = numpy.where(extended_input_mask == clazz['val'], clazz['color'], self.input_img)
        # self.mask = numpy.where(self.input_mask != UNKNOWN['val'], self.input_mask, self.mask)

        cv2.imshow(self.input_window_name, self.input_img)
        cv2.imshow(self.output_window_name, self.output_img)


def run():
    length = 25
    array = get_random_array(length)
    print('Initialize state')
    print(array)

    merge_sort(array)

    print('After sort state')
    print(array)


def get_random_array(length):
    array = []
    for _ in range(length):
        array.append(random.random())
    return array


def merge_sort(array):
    length = len(array)
    merge_sort_internal(array, 0, length)


def merge_sort_internal(array, begin, end):
    length = end - begin
    if length < 2:  # array of length 0 or 1 is always sorted
        return
    half_length = length / 2
    merge_sort_internal(array, begin, begin + half_length)
    merge_sort_internal(array, begin + half_length, end)

    temp_array = []
    i = begin
    j = begin + half_length

    while i < begin + half_length and j < end:
        a = array[i]
        b = array[j]
        if a < b:
            temp_array.append(a)
            i += 1
        else:
            temp_array.append(b)
            j += 1
    while i < begin + half_length:
        temp_array.append(array[i])
        i += 1
    while j < end:
        temp_array.append(array[j])
        j += 1

    # Copy sorted items from temp array
    for i in range(begin, end):
        array[i] = temp_array[i]
