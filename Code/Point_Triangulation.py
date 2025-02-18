import numpy as np
import cv2

# A function, for triangulation, given the image pair and their corresponding projection matrices
def Triangulation(P1, P2, pts1, pts2, K, repeat):
    if not repeat:
        points1 = np.transpose(pts1)
        points2 = np.transpose(pts2)
    else:
        points1 = pts1
        points2 = pts2
    # print("points1", points1.shape, "points2", points2.shape)
    # print("P1", P1.shape, "P2", P2.shape)
    cloud = cv2.triangulatePoints(P1, P2, points1, points2)
    cloud = cloud / cloud[3]

    return points1, points2, cloud