import cv2
import numpy as np

# A function, to downscale the image in case SfM pipeline takes time to execute.
def img_downscale(img, downscale):
	downscale = int(downscale/2)
	i = 1
	while(i <= downscale):
		img = cv2.pyrDown(img)
		i = i + 1
	return img
	
# Feature detection for two images, followed by feature matching
def find_features(img0, img1):
    img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # plt.imshow(img0gray)
    # plt.imshow(img1gray)
    # plt.pause(2)
    sift = cv2.xfeatures2d.SIFT_create()
    kp0, des0 = sift.detectAndCompute(img0gray, None)
    
    #lk_params = dict(winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    #pts0 = np.float32([m.pt for m in kp0])
    # pts1, st, err = cv2.calcOpticalFlowPyrLK(img0gray, img1gray, pts0, None, **lk_params)
    #pts0 = pts0[st.ravel() == 1]
    #pts1 = pts1[st.ravel() == 1]
    # print(pts0.shape, pts1.shape)

    # print(len(kp0), len(kp1))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    print("imgpts1", pts0.shape, "imgpts2", pts1.shape)
    # matched_image = cv2.drawMatches(img0, kp0, img1, kp1, good, None,
    #                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.namedWindow('matched_image', cv2.WINDOW_NORMAL)
    # cv2.imshow("matched_image", matched_image)

    return pts0, pts1