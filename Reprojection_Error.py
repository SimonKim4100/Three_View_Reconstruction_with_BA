import numpy as np
import cv2

# Calculation for Reprojection error in main pipeline
def ReprojectionError(X, pts, Rt, K, homogenity):
    total_error = 0
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    r, _ = cv2.Rodrigues(R)
    
    # if homogenity == 0:
    #     X = cv2.convertPointsFromHomogeneous(X)

    p, _ = cv2.projectPoints(X, r, t, K, distCoeffs=None)
    p = p[:, 0, :]
    p = np.float32(p)
    pts = np.float32(pts)

    total_error = cv2.norm(p, pts, cv2.NORM_L2) 
    tot_error = total_error / len(p)

    return tot_error, X, p

# Calculation of reprojection error for bundle adjustment
def OptimReprojectionError(x, n2d):
	Rt = x[0:12].reshape((3,4))
	K = x[12:21].reshape((3,3))
	p = x[21:21 + n2d].reshape((2, int(n2d/2))).T
	X = x[21 + n2d:].reshape((int(len(x[21 + n2d:])/3), 3))
	R = Rt[:3, :3]
	t = Rt[:3, 3]

	num_pts = len(p)
	error = []
	r, _ = cv2.Rodrigues(R)

	p2d, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	p2d = p2d[:, 0, :]
	for idx in range(num_pts):
		img_pt = p[idx]
		reprojected_pt = p2d[idx]
		er = img_pt - reprojected_pt
		error.append(er)

	err_arr = np.array(error).ravel()

	return err_arr