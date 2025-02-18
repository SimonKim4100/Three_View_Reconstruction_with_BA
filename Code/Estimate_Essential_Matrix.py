import cv2
import numpy as np

def find_Essential_Matrix(pts0, pts1, K):

    # Finding essential matrix
    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]
    # print('shape1', pts0.shape, pts1.shape)
    # The pose obtained is for second image with respect to first image
    _, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)  # |finding the pose
    # print("before_mask", t.shape)
    pts0 = pts0[mask.ravel() > 0]
    pts1 = pts1[mask.ravel() > 0]

    return R, t, pts0, pts1