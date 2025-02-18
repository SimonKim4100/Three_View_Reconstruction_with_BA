import numpy as np
import cv2

# Pipeline for Perspective - n - Point
def PnP(X, p, K, d, p_0, initial):
    # print(X.shape, p.shape, p_0.shape)
    if initial == 1:
        X = X[:, 0, :]
        p = p.T
        p_0 = p_0.T

    ret, rvecs, t, inliers = cv2.solvePnPRansac(X, p, K, d, cv2.SOLVEPNP_ITERATIVE)
    # print(X.shape, p.shape, t, rvecs)
    R, _ = cv2.Rodrigues(rvecs)

    if inliers is not None:
        p = p[inliers[:, 0]]
        X = X[inliers[:, 0]]
        p_0 = p_0[inliers[:, 0]]

    return R, t, p, X, p_0
