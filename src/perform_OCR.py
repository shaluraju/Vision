
import pytesseract
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
#from skimage.filters import threshold_local
#print(np.__version__)

def plt_imshow(title, image):
	# convert the image frame BGR to RGB color space and display it
    #plt.figure()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # return the warped image
    return warped

#----------------------   Reading Image -------------------------------

configuration = "--psm 1"
image = cv2.imread('/home/shalu/catkin_ws/src/character_recognition/images/compartment.png')


#print(image)
#print(image.shape)
#
#print(image.dtype)


threshold = 80

# Converting into Binary(0 to 1)
image_B = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)

image_B = cv2.GaussianBlur(image_B,(5,5), 0)

edged = cv2.Canny(image_B, threshold, 200)
#cv2.imshow("edged", edged)
#plt_imshow("Edged", edged)


# ----------------- Contour----------------------------

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
    
#print(screenCnt)
contour_image = cv2.drawContours(image, [screenCnt], -1, (255, 0, 0), 4)
#plt_imshow("Contour Identified", contour_image)

#-----------------   Crop  ----------------------------

#cropped_image = edged[]

top_left = [min(screenCnt[2][0][0], screenCnt[3][0][0]), min(screenCnt[2][0][1], screenCnt[3][0][1])]
top_right = [max(screenCnt[2][0][0], screenCnt[3][0][0]), max(screenCnt[2][0][1], screenCnt[3][0][1])]

bot_right = [min(screenCnt[0][0][0], screenCnt[1][0][0]), max(screenCnt[0][0][1], screenCnt[1][0][1])]
bot_left= [min(screenCnt[0][0][0], screenCnt[1][0][0]), min(screenCnt[0][0][1], screenCnt[1][0][1])]
#print("top_left: ", top_left)

#print("top_rit: ", top_right)
#print("bot left: ", bot_left)
#print("bot_rigt: ", bot_right)

cropped_img = image_B[bot_left[1] - 10: bot_right[1] + 10 , top_left[1] - 10:top_right[0] + 10]
#plt_imshow("Crop", cropped_img)

#cv2.drawContours(image, [top_left top_right bot_left bot_right], -1, (255, 0, 0), 4)
#plt_imshow("Cropped: ", cropped_img)


#----------   OCR   ----------
thresh = cv2.threshold(cropped_img, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
plt_imshow("otsu", thresh)       

adp_thre = cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,11)
plt_imshow("adaptive", adp_thre)   


adp_thre_erode = cv2.erode(adp_thre, None, 15)
plt_imshow("Erode", adp_thre_erode) 

text = pytesseract.image_to_string(adp_thre)
print(text)

tophat = cv2.morphologyEx(adp_thre_erode, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25,4)))
plt_imshow("Top Hat", tophat)

text = pytesseract.image_to_string(tophat)
print(text)



