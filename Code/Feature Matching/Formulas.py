import numpy as np
import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread('.../queryImage.png',0) # queryImage
img1 = cv2.imread('.../trainImage.png',0) # trainImage
print(img2.shape)
print(img1.shape)

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

good_matches = matches[:30]

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches     ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

dst = cv2.perspectiveTransform(pts,M)
dst += (w, 0)  # adding offset

draw_params = dict(matchColor = (255,0,0), # draw matches in green color
               singlePointColor = None,
               matchesMask = matchesMask, # draw only inliers
               flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None,**draw_params)

# =============================================================================
img3 = cv2.polylines(img3,[np.int32(dst)],True,255,3, cv2.LINE_AA)
cv2.imshow("result", img3)
cv2.waitKey()
# # # print(np.int32(dst))
# # # #print(np.int32(dst[0][0])[0])
# # # #print(np.int32(dst[1][0])[0])
# # # #print(np.int32(dst[1][0])[1])
# # # #print(np.int32(dst[2][0])[1])

CenterX = np.int32(dst[0][0])[0] + 1/2*(np.int32(dst[1][0])[0] - np.int32(dst[0][0])[0])
CenterY = np.int32(dst[2][0])[1] + 1/2*(np.int32(dst[1][0])[1] - np.int32(dst[2][0])[1])
print(CenterX)
print(CenterY)
print(len(img3))
print(len(img3[0]))

temp1 = -6.1946 - (-6.0981)
TempX = CenterX*(temp1)/len(img3[0])
#print(TempX)
X = -6.1946 - TempX
print(X)

temp2 = 56.6081 - 56.5747
TempY = CenterY*(temp2)/len(img3)
print(TempY)
Y = 56.5747 + TempY
print(Y)

