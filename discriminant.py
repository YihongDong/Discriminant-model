def imCalibration(baseIm , targetIm):
    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt
    
    [height,width,depth] = targetIm.shape
    [height2,width2,depth] = baseIm.shape
    
    top, bot, left, right =  200 , 200, 200, 200
    
    srcImg = cv.copyMakeBorder(targetIm, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv.copyMakeBorder(baseIm, top, bot + (height - height2), left, right + (width - width2), cv.BORDER_CONSTANT, value=(0, 0, 0))
    
    grayIm = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    targetImgray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)
    
    sift = cv.xfeatures2d_SIFT().create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(grayIm, None)
    kp2, des2 = sift.detectAndCompute(targetImgray, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=matchesMask,
    #                    flags=0)
    #img3 = cv.drawMatchesKnn(grayIm, kp1, targetImgray, kp2, matches, None, **draw_params)
    #plt.imshow(img3, ), plt.show()

    rows, cols = srcImg.shape[:2]
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]), flags=cv.WARP_INVERSE_MAP)

        # for col in range(0, cols):
        #     if srcImg[:, col].any() and warpImg[:, col].any():
        #         tleft = col
        #         break
        # for col in range(cols-1, 0, -1):
        #     if srcImg[:, col].any() and warpImg[:, col].any():
        #         tright = col
        #         break

#        res = np.zeros([rows, cols, 3], np.uint8)
#        for row in range(0, rows):
#            for col in range(0, cols):
#                if srcImg[row, col].any():
#                    res[row, col] = srcImg[row, col]
#                if warpImg[row, col].any() and srcImg[row, col].any():
#                    res[row, col] = warpImg[row, col]

        # opencv is bgr, matplotlib is rgb
        
        mergedIm = warpImg[top: (height+top), left: (width+left)]#res[top: (height+top), left: (width+left)]
        #cv.imwrite('mergedIm.jpg',mergedIm)
        #cv.imwrite('res.jpg',res)
        #cv.imwrite('warpImg.jpg',warpImg)
        #mergedIm = cv.cvtColor(res, cv.COLOR_BGR2RGB)
        # show the result
        #plt.figure()
        #plt.imshow(res)
        #plt.show()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        mergedIm = targetIm
        
    return mergedIm



def anomalyDetection(baseIm,targetIm, threshold = 40, contourAera = 100):

    from skimage.measure import compare_ssim
    import imutils
    import cv2
    from matplotlib import pyplot as plt


    calibratedIm = imCalibration(baseIm,targetIm)

    grayA = cv2.cvtColor(targetIm, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(calibratedIm, cv2.COLOR_BGR2GRAY)

    #grayA = RminusB(targetIm)
    #grayB = RminusB(calibratedIm)

    #cv.imwrite("grayA.jpg",grayA)
    #cv.imwrite("grayB.jpg",grayB)

    (score,diff) = compare_ssim(grayA,grayB,full = True)
    diff = (diff *255).astype("uint8")
    #print("SSIM:{}".format(score))


#    blur = cv2.GaussianBlur(diff,(5,5),0)
    thresh = cv2.threshold(diff,(1.1 - score) * 255,255,cv2.THRESH_TOZERO_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations=2)
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations=2)


    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    (x_f, y_f, w_f, h_f) = ([0,0], [0,0], [0,0], [0,0])
    maxarea = [0, 0]
    [height, width, depth]=targetIm.shape
    for c in cnts:
        if(cv2.contourArea(c)>contourAera) :
            (x,y,w,h) = cv2.boundingRect(c)
            # cv2.rectangle(targetIm, (x,y),(x+w,y+h), (0, 0, 255), 2)
            # cv2.rectangle(calibratedIm,(x,y),(x+w,y+h),(0,0,255),2)
            if w < width*0.05 or h < height*0.05:
                continue
            if w*h > maxarea[0]:
                maxarea[0] = w*h
                (x_f[0], y_f[0], w_f[0], h_f[0]) = (x,y,w,h)
            elif w*h <= maxarea[0] and w*h > maxarea[1]:
                maxarea[1] = w * h
                (x_f[1], y_f[1], w_f[1], h_f[1]) = (x, y, w, h)

    cv2.rectangle(targetIm,(x_f[0],y_f[0]),(x_f[0]+w_f[0],y_f[0]+h_f[0]),(0,0,255),2)
    if w_f[1]*h_f[1] > 0.6*w_f[0]* h_f[0]:
        cv2.rectangle(targetIm, (x_f[1], y_f[1]), (x_f[1] + w_f[1], y_f[1] + h_f[1]), (0, 0, 255), 2)

    return targetIm

def rescale(img):
    import numpy as np
    area = 1000000
    [height,width,depth] = img.shape
    scale = np.sqrt(area/(height*width))
    width = int(width*scale)
    height = int(height*scale)

    return width,height,scale

import cv2 as cv
from matplotlib import pyplot as plt
import time

start = time.time()
baseIm = cv.imread('0002_normal.jpg')
#width,height,scale = rescale(baseIm)
#baseIm = cv.resize(baseIm, (width, height), interpolation=cv.INTER_AREA)


targetIm = cv.imread('0002_3.jpg')
#width,height,scale = rescale(targetIm)
#targetIm = cv.resize(targetIm, (width, height), interpolation=cv.INTER_AREA)


markedIm =  anomalyDetection(baseIm,targetIm, 50,300)
print(time.time() - start)


plt.figure()
plt.imshow(cv.cvtColor(baseIm, cv.COLOR_BGR2RGB))

plt.figure()
plt.imshow(cv.cvtColor(markedIm, cv.COLOR_BGR2RGB))

cv.imwrite("Modified.jpg", markedIm)



