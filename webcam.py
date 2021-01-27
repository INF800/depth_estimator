import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.filters import threshold_local
import imutils


def disp_for(im, delay=0.000001):
    plt.axis('off')
    plt.imshow(im)
    plt.show(block=False)
    plt.pause(delay)
    plt.close()


def get_contour_bbox(image):
    
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    # edge det
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    #edged = cv2.Canny(gray, 30, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    
    if screenCnt is not None:
        b, a, d, c = screenCnt
        #print(b[0], '>>', b[0][0], '>>', b[0][1])
        width = b[0][0] - a[0][0]
        cv2.putText(image, f'pixel width: {width}', (b[0][0], b[0][1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0,244, 255),1) 
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    return image


def workon(frame):
    frame = get_contour_bbox(frame)
    disp_for(frame, 0.00001)




cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    #workon(cv2.imread('images/find_my_contour.png'))
    workon(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()