from skimage.filters import threshold_local
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils

"""
read image
"""

from config import INPUT_PATH
image = cv2.imread( str(INPUT_PATH) )
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)


"""
edge detection
"""

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

#plt.imshow(edged)
#plt.show()


"""
finding contours
"""

contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
for c in contours:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

# idxs in `approx`
# 1    0
# +----+
# |    |
# +----+
# 2    3
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)


#plt.imshow(image)
#plt.show()
