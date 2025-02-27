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
def sift(img0, img1):
    img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(contrastThreshold=0.1, edgeThreshold=30)
    kp0, des0 = sift.detectAndCompute(img0gray, None)
    kp1, des1 = sift.detectAndCompute(img1gray, None)

    # Lucas-Kanade method: Only for small movement(ie, consecutive frame of vid)
    #pts0 = np.float32([m.pt for m in kp0])
    # pts1, st, err = cv2.calcOpticalFlowPyrLK(img0gray, img1gray, pts0, None, **lk_params)
    #pts0 = pts0[st.ravel() == 1]
    #pts1 = pts1[st.ravel() == 1]

    # Brute force
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    # Lowe's Ratio Test
    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    # correspondence = cv2.drawMatches(img0, kp0, img1, kp1, good, None,
    #                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.namedWindow('Correspondence', cv2.WINDOW_NORMAL)
    # cv2.imshow("Correspondence", correspondence)

    return pts0, pts1

def Find_Essential_Matrix(pts0, pts1, K):

    # Finding essential matrix
    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=0.5, mask=None)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]
    # Find extrinsic matrix
    _, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)
    pts0 = pts0[mask.ravel() > 0]
    pts1 = pts1[mask.ravel() > 0]
    # Scaline issue
    # t *= np.median(np.linalg.norm(pts0 - pts1, axis=1))

    return R, t, pts0, pts1

def Triangulation(P1, P2, pts1, pts2):
    points1 = np.transpose(pts1)
    points2 = np.transpose(pts2)
    cloud = cv2.triangulatePoints(P1, P2, points1, points2)
    cloud = np.transpose(cloud)
    # Return to Euclidean coords
    cloud = cv2.convertPointsFromHomogeneous(cloud)[:, 0, :]
    return pts1, pts2, cloud

def PnP(X, p, K, d, p_0, initial):
    # if initial == 0:
    #     X = X[:, 0, :]
    #     p = p.T
    #     p_0 = p_0.T

    _, rvecs, t, inliers = cv2.solvePnPRansac(X, p, K, d, cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvecs)

    if inliers is not None:
        p = p[inliers[:, 0]]
        X = X[inliers[:, 0]]
        p_0 = p_0[inliers[:, 0]]

    return R, t, p, X, p_0

# def draw_optical_flow(img1, img2, pts1, pts2, output_path=None):
#     # Convert to BGR if the image is grayscale
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
#     # Create a copy of the image to draw arrows
#     img_with_arrows = img.copy()

#     # Draw arrows for each corresponding pair
#     for (x1, y1), (x2, y2) in zip(pts1, pts2):
#         cv2.arrowedLine(img_with_arrows, 
#                         (int(x1), int(y1)), (int(x2), int(y2)), 
#                         color=(0, 255, 0), thickness=1, tipLength=0.1)

#     # Show the result
#     cv2.imshow("Optical Flow", img_with_arrows)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Save the image if output_path is provided
#     if output_path:
#         cv2.imwrite(output_path, img_with_arrows)

#     return img_with_arrows

def draw_optical_flow(img0, img1, pts0, pts1, output_path=None):
    color=(0, 255, 0)
    thickness=1
    # Convert images to color (if grayscale)
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    
    # Get image dimensions
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Concatenate images side by side
    height = max(h0, h1)
    width = w0 + w1
    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    output_img[:h0, :w0] = img0
    output_img[:h1, w0:w0 + w1] = img1

    # Draw lines between matching keypoints
    for pt0, pt1 in zip(pts0, pts1):
        pt0 = tuple(map(int, pt0))  # Keypoint in first image (x, y)
        pt1 = tuple(map(int, pt1))  # Keypoint in second image (x, y)

        # Shift pt1's x-coordinate since second image is placed to the right
        pt1 = (pt1[0] + w0, pt1[1])

        # Draw line connecting the points
        cv2.line(output_img, pt0, pt1, color, thickness, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, output_img)

    return output_img