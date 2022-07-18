import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys, subprocess, cv2
from PIL import Image, ImageDraw

MIN_MATCH_COUNT = 10 #Number of minimum required feature points
good_match_rate = 0.7

img2 = cv2.imread('testaki.png',0) # queryImage
img1 = cv2.imread('s1.png',0) # trainImage
 
# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFmatcher object
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
# Match descriptors.
matches = bf.match(des1,des2)


matches = sorted(matches, key = lambda x:x.distance)
print(len(matches))
good = matches[:int(len(matches) * good_match_rate)] 
print(good)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    #print ("Not enough matches are found - %d/%d") % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

########################################################################################################
image_size = img1.shape[:2]
template_size = img2.shape[:2]
print ('image_size (y,x)', image_size)
print ('template_size (y,x)', template_size)
#cv2.MatchTemplate(im, tmp, result, cv2.CV_TM_SQDIFF)
result = cv2.matchTemplate(img1, img2, cv2.TM_SQDIFF)

#print ('DEBUG:result', result)


#print ('\nSTEP 5: Get the Min Max Loc')

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#print ('\n')

#print ('result', result)
print ('min_val', min_val)
print ('max_val', max_val)
print ('min_loc', min_loc, 'X')
print ('max_loc', max_loc)

confidence = (9999999999 - min_val) / 100000000
print ('primary confidence', '%.2f %%' % confidence)

altconfidence = 100 - ((min_val / max_val)*100)
print ('alternate confidence', '%.2f %%' % altconfidence)

topleftx = min_loc[0]
toplefty = min_loc[1]
sizex = template_size[1]
sizey = template_size[0]

if ((confidence > 90) or (altconfidence > 80) and (confidence > 95) ) or (altconfidence > 99) or ((confidence > 97) and (altconfidence > 93)) or ((confidence > 95.7) and (altconfidence > 96.3)):
  print ('The image of size', template_size, '(y,x) was found at', min_loc)
  print ('Marking', 'm1.png', 'with a red rectangle')
  marked = Image.open('testaki.png')
  draw = ImageDraw.Draw(marked)
  draw.line(((topleftx,         toplefty),         (topleftx + sizex, toplefty)),           fill="red", width=2)
  draw.line(((topleftx + sizex, toplefty),         (topleftx + sizex, toplefty + sizey)),   fill="red", width=2)
  draw.line(((topleftx + sizex, toplefty + sizey), (topleftx,         toplefty + sizey)),   fill="red", width=2)
  draw.line(((topleftx,         toplefty + sizey), (topleftx,         toplefty)),           fill="red", width=2)
  del draw 
  marked.save('result.png', "PNG")


########################################################################################################
img1 = cv2.imread('testaki.png',0)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)


plt.imshow(img3, 'gray'),plt.show()

cv2.imwrite('results2.png', img3)