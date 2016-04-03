"""
    Segmentation of images to four segments.
    Usage: python <>.py <input-image-path> <path-for-output-segmented-image> <path-for-output-mask>
    Example: python __main__.py ../img/man.jpg ./output/segmented.jpg ./output/mask.txt

    If the input path is a valid image, you will be able to mark four segments on the image.
    Key presses:
    0-3     - changes the segment number currently being marked
    n       - uses the current markings to calculate the segments and display them in the output window
    s       - save the output image and mask to the given output paths
"""

import sys
import segmentation
import cv2

print(__doc__)
if len(sys.argv) != 4:
    print('The number of command-line parameters is incorrect.')
    exit(1)
input_path = sys.argv[1]
if cv2.imread(input_path) is None:
    print('Invalid input file format')
    exit(1)
output_image_path = sys.argv[2]
output_mask_path = sys.argv[3]
try:
    f = open(output_image_path, 'w')
    f.close()
    f = open(output_mask_path, 'w')
    f.close()
except IOError:
    print('The folder holding the output files should exist.')
    exit(1)

segmentor = segmentation.ImageSegmentation(input_path, output_image_path, output_mask_path)
segmentor.run()
