import numpy as np
import cv2

RED = np.array([0, 0, 255], dtype=np.uint8)
BLUE = np.array([255, 0, 0], dtype=np.uint8)
GREEN = np.array([0, 255, 0], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)

UNKNOWN_AREA = np.uint8(-1)  # an area not marked by the user

CLASS_COLORS = [BLACK, WHITE, BLUE, GREEN]
RECT_COLOR = RED

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

        # Grabcut Parameters - phases 1 2 3
        self.grabcut_params = [{}, {}, {}]

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
            rect_bold = 2

            # Draw Rectangle
            if event == cv2.EVENT_RBUTTONDOWN:
                self.rect_drawn = True
                self.rect_start_x, self.rect_start_y = x, y
                self.input_img = self.original_img.copy()
                start_point = (self.rect[0], self.rect[1])
                end_point = (self.rect[2], self.rect[3])
                cv2.rectangle(self.input_img, start_point, end_point, RECT_COLOR, rect_bold)
                self.update_windows()

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.rect_drawn:
                    self.rect = (min(self.rect_start_x, x), min(self.rect_start_y, y), max(self.rect_start_x, x),
                                 max(self.rect_start_y, y))
                    self.input_img = self.original_img.copy()
                    start_point = (self.rect[0], self.rect[1])
                    end_point = (self.rect[2], self.rect[3])
                    cv2.rectangle(self.input_img, start_point, end_point, RECT_COLOR, rect_bold)
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
                    cv2.rectangle(self.input_img, start_point, end_point, RECT_COLOR, rect_bold)
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

        if not self.is_gc_initiated:
            for phase in self.grabcut_params:
                phase['mask'] = np.full(self.input_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
                # phase['mask'] = np.random.choice([cv2.GC_PR_BGD, cv2.GC_PR_FGD], size=self.input_mask.shape).astype(
                #     np.uint8)
                phase['fgdmodel'] = None
                phase['bgdmodel'] = None
            self.is_gc_initiated = True

        # Phase 1

        gc_phase = self.grabcut_params[0]
        gc_phase['mask'] = np.where(self.input_mask == 0, cv2.GC_FGD, gc_phase['mask'])
        gc_phase['mask'] = np.where(self.input_mask == 1, cv2.GC_FGD, gc_phase['mask'])
        gc_phase['mask'] = np.where(self.input_mask == 2, cv2.GC_BGD, gc_phase['mask'])
        gc_phase['mask'] = np.where(self.input_mask == 3, cv2.GC_BGD, gc_phase['mask'])

        try:
            if not self.is_gc_initiated:
                cv2.grabCut(self.original_img, gc_phase['mask'], None, gc_phase['bgdmodel'], gc_phase['fgdmodel'], 3,
                            cv2.GC_INIT_WITH_MASK)
            else:
                print gc_phase['mask'].sum()
                cv2.grabCut(self.original_img, gc_phase['mask'], None, gc_phase['bgdmodel'], gc_phase['fgdmodel'], 3,
                            cv2.GC_INIT_WITH_MASK)
                print gc_phase['mask'].sum()
        except cv2.error:
            print('GrabCut failed. There may be not enough information from the user')

        classes_0_1 = np.logical_or(gc_phase['mask'] == cv2.GC_PR_FGD, gc_phase['mask'] == cv2.GC_FGD)
        classes_0_1 = classes_0_1.astype('uint8')
        classes_2_3 = np.logical_not(classes_0_1)

        # Phases 2+3

        marked_classes = np.unique(self.input_mask)
        class_0 = np.full(self.input_mask.shape, fill_value=False)
        class_1 = np.full(self.input_mask.shape, fill_value=False)
        class_2 = np.full(self.input_mask.shape, fill_value=False)
        class_3 = np.full(self.input_mask.shape, fill_value=False)

        # Phase 2

        if 0 in marked_classes and 1 in marked_classes:
            gc_phase = self.grabcut_params[1]
            gc_phase['mask'] = np.full(self.input_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
            gc_phase['mask'] = np.where(self.input_mask == 0, cv2.GC_FGD, gc_phase['mask'])
            gc_phase['mask'] = np.where(self.input_mask == 1, cv2.GC_BGD, gc_phase['mask'])
            try:
                if not self.is_gc_initiated:
                    cv2.grabCut(self.original_img, gc_phase['mask'], None, gc_phase['bgdmodel'], gc_phase['fgdmodel'], 3,
                                cv2.GC_INIT_WITH_MASK)
                else:
                    cv2.grabCut(self.original_img, gc_phase['mask'], None, gc_phase['bgdmodel'], gc_phase['fgdmodel'], 3,
                                cv2.GC_EVAL)
            except cv2.error:
                print('GrabCut failed. There may be not enough information from the user')

            class_0 = np.logical_or(gc_phase['mask'] == cv2.GC_PR_FGD, gc_phase['mask'] == cv2.GC_FGD)
            class_0 = np.logical_and(class_0, classes_0_1)
            class_1 = np.logical_and(classes_0_1, np.logical_not(class_0))
        elif 0 in marked_classes:
            class_0 = classes_0_1
        elif 1 in marked_classes:
            class_1 = classes_0_1

        # Phase 3

        if 2 in marked_classes and 3 in marked_classes:
            gc_phase = self.grabcut_params[2]
            gc_phase['mask'] = np.full(self.input_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
            gc_phase['mask'] = np.where(self.input_mask == 2, cv2.GC_FGD, gc_phase['mask'])
            gc_phase['mask'] = np.where(self.input_mask == 3, cv2.GC_BGD, gc_phase['mask'])
            try:
                if not self.is_gc_initiated:
                    cv2.grabCut(self.original_img, gc_phase['mask'], None, gc_phase['bgdmodel'], gc_phase['fgdmodel'], 3,
                                cv2.GC_INIT_WITH_MASK)
                else:
                    cv2.grabCut(self.original_img, gc_phase['mask'], None, gc_phase['bgdmodel'], gc_phase['fgdmodel'], 3,
                                cv2.GC_EVAL)
            except cv2.error:
                print('GrabCut failed. There may be not enough information from the user')

            class_2 = np.logical_or(gc_phase['mask'] == cv2.GC_PR_FGD, gc_phase['mask'] == cv2.GC_FGD)
            class_2 = np.logical_and(class_2, classes_2_3)
            class_3 = np.logical_and(classes_2_3, np.logical_not(class_2))
        elif 2 in marked_classes:
            class_2 = classes_2_3
        elif 3 in marked_classes:
            class_3 = classes_2_3

        self.output_img = np.ndarray(self.original_img.shape, dtype=np.uint8)
        self.output_img = np.where(triple(class_0), CLASS_COLORS[0], self.output_img)
        self.output_img = np.where(triple(class_1), CLASS_COLORS[1], self.output_img)
        self.output_img = np.where(triple(class_2), CLASS_COLORS[2], self.output_img)
        self.output_img = np.where(triple(class_3), CLASS_COLORS[3], self.output_img)

    def update_windows(self):
        cv2.imshow(self.input_window_name, self.input_img)
        cv2.imshow(self.output_window_name, self.output_img)


def triple(matrix):
    return np.dstack((matrix, matrix, matrix))
