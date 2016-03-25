import numpy as np
import cv2

RED = np.array([0, 0, 255], dtype=np.uint8)
BLUE = np.array([255, 0, 0], dtype=np.uint8)
GREEN = np.array([0, 255, 0], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)

UNKNOWN_AREA = np.uint8(-1)  # an area not marked by the user

CLASS_COLORS = [BLACK, WHITE, RED, GREEN]

ESCAPE_KEY = 27


class ImageSegmentation:
    def __init__(self, file_name):
        self.input_window_name = 'input'
        self.output_window_name = 'output'

        self.original_img = cv2.imread(file_name)
        self.input_img = self.original_img.copy()
        self.output_img = np.zeros(self.input_img.shape, np.uint8)

        #  Rectangle
        self.rect = (0, 0, 0, 0)
        self.rect_drawn = False
        self.rect_ready = False
        self.rect_start_x = 0
        self.rect_start_y = 0

        #  Classes
        self.line_class = 0
        self.line_drawn = False
        self.input_mask = np.zeros(self.original_img.shape[:2], np.uint8)
        self.input_mask += UNKNOWN_AREA

        # Mask
        self.mask = np.full(self.original_img.shape[:2], cv2.GC_PR_BGD, np.uint8)

        # Arrays for GrabCut
        self.bgdmodel = None
        self.fgdmodel = None

        # Hash grabcut been initiated
        self.is_gc_initiated = False

        cv2.namedWindow(self.input_window_name)
        cv2.namedWindow(self.output_window_name)
        cv2.setMouseCallback(self.input_window_name, self.create_mouse_listener())

    def run(self):
        while True:
            self.update_windows()
            key = cv2.waitKey()

            if key == ESCAPE_KEY:  # esc  exit
                break
            elif key == ord('0'):
                self.line_class = 0
            elif key == ord('1'):
                self.line_class = 1
            elif key == ord('2'):
                self.line_class = 2
            elif key == ord('3'):
                self.line_class = 3
            elif key == ord('n'):
                self.calculate_cut()
            print('choose ' + str(unichr(key)))
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

            elif event == cv2.EVENT_RBUTTONUP:
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

            # Color Lines
            if event == cv2.EVENT_LBUTTONDOWN:
                self.line_drawn = True
                clazz_color = CLASS_COLORS[self.line_class]
                cv2.circle(self.input_mask, (x, y), thickness, self.line_class, -1)
                cv2.circle(self.input_img, (x, y), thickness, clazz_color.tolist(), -1)
                self.update_windows()

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.line_drawn:
                    clazz_color = CLASS_COLORS[self.line_class]
                    cv2.circle(self.input_mask, (x, y), thickness, self.line_class, -1)
                    cv2.circle(self.input_img, (x, y), thickness, clazz_color.tolist(), -1)
                    self.update_windows()

            elif event == cv2.EVENT_LBUTTONUP:
                if self.line_drawn:
                    self.line_drawn = False
                    clazz_color = CLASS_COLORS[self.line_class]
                    cv2.circle(self.input_mask, (x, y), thickness, self.line_class, -1)
                    cv2.circle(self.input_img, (x, y), thickness, clazz_color.tolist(), -1)
                    self.update_windows()

        return mouse_listener

    def calculate_cut(self):
        if len(np.unique(self.input_mask)) <= 2:
            print('Put some colors on the picture!!')
            return

        self.mask = np.where(self.input_mask == 0, cv2.GC_BGD, self.mask)
        self.mask = np.where(self.input_mask == 1, cv2.GC_BGD, self.mask)
        self.mask = np.where(self.input_mask == 2, cv2.GC_FGD, self.mask)
        self.mask = np.where(self.input_mask == 3, cv2.GC_FGD, self.mask)

        try:
            if not self.is_gc_initiated:
                cv2.grabCut(self.original_img, self.mask, None, self.bgdmodel, self.fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
            else:
                cv2.grabCut(self.original_img, self.mask, None, self.bgdmodel, self.fgdmodel, 1, cv2.GC_EVAL)
        except cv2.error:
            print('GrabCut failed. There may be not enough information from the user')

        classes_1_2 = np.logical_or(self.mask == cv2.GC_PR_FGD, self.mask == cv2.GC_FGD) * np.array([255])
        classes_1_2 = classes_1_2.astype('uint8')
        self.output_img = cv2.bitwise_and(self.original_img, self.original_img, mask=classes_1_2)

    def update_windows(self):
        cv2.imshow(self.input_window_name, self.input_img)
        cv2.imshow(self.output_window_name, self.output_img)
