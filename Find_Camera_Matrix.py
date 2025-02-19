import numpy as np
import cv2
import glob
import os

def Find_Camera_Matrix():
    size = (7, 11)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((np.prod(size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

    objpoints = [] # real world space
    imgpoints = [] # image plane

    images = glob.glob('Checkerboard/*.jpg')
    images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    for _, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(gray, size, None)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (23,23),(-1,-1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera Matrix: \n", mtx)

    return mtx, dist, rvecs, tvecs