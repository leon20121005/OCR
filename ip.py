import cv2
import numpy as np

image = cv2.imread("example2.jpg")

FIRST_THRESHOLDING_THRESHOLD = 100
DILATION_ITERATION = 3
# example.jpg = 380, 150
MAX_AREA = 180
MIN_AREA = 70

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# threshold (inverse)
_, thresh = cv2.threshold(gray, FIRST_THRESHOLDING_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# dilate
dilated = cv2.dilate(thresh,kernel,iterations = DILATION_ITERATION)
# get contours
_, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# define pts_dst
_width  = 600.0
_height = 420.0
_margin = 0.0
corners = np.array(
    [
	    [[  		_margin, _margin 		   ]],
		[[ 			_margin, _height + _margin ]],
		[[ _width + _margin, _height + _margin ]],
		[[ _width + _margin, _margin 		   ]],
	]
)
pts_dst = np.array(corners, np.float32)

index = 1

# for each contour found, process it and draw a rectangle around it on original image
for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h > MAX_AREA and w > MAX_AREA:
        continue

    # discard areas that are too small
    if h < MIN_AREA or w < MIN_AREA:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

    arc_len = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.1 * arc_len, True)

    if (len(approx) == 4):
        pts_src = np.array(approx, np.float32)
        h, status = cv2.findHomography(pts_src, pts_dst)
        out = cv2.warpPerspective(image, h, (int(_width + _margin * 2), int(_height + _margin * 2)))

        # output raw seperated image
        # cv2.imwrite("raw_input" + str(index) + ".jpg", out)

        # replace red element with white element
        out_hsv=cv2.cvtColor(out, cv2.COLOR_BGR2HSV)

        # lower mask (0-10)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(out_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(out_hsv, lower_red, upper_red)

        # join my masks
        mask = mask0 + mask1

        # set my output img to zero everywhere except my mask
        red_removed_out = out.copy()
        red_removed_out[np.where(mask)] = [255, 255, 255]

        # output red removed seperated image
        # cv2.imwrite("red_removed_input" + str(index) + ".jpg", red_removed_out)

        # thresholding
        retval, threshold_out = cv2.threshold(red_removed_out, 100, 255, cv2.THRESH_BINARY_INV)
        # output thresholding seperated image
        # cv2.imwrite("threshold_input" + str(index) + ".jpg", threshold_out)

        # resizing
        output_image = cv2.resize(threshold_out, (20, 20))

        cv2.imwrite("input" + str(index) + ".jpg", output_image)
        index += 1

# output gray image
# cv2.imwrite("gray.jpg", gray)

# output first thresholding image
# cv2.imwrite("thresh.jpg", thresh)

# output dilated image
# cv2.imwrite("dilated.jpg", dilated)

# output edges image
edges = cv2.Canny(gray, 10, 250)
cv2.imwrite("edges.jpg", edges)

# write original image with added contours to disk
cv2.imwrite("contoured.jpg", image)