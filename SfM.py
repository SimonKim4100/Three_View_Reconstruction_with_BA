import cv2
import numpy as np
import os

from Find_Camera_Matrix import Find_Camera_Matrix
from Features import img_downscale, sift, Find_Essential_Matrix, Triangulation, PnP, draw_optical_flow
from Reprojection_Error import ReprojectionError
from Bundle_Adjustment import BundleAdjustment
from Point_Cloud_Formation import view_point_cloud, common_points

# Current Path Directory
path = os.getcwd()

img_dir = os.path.join(path, "Images")
# A provision for bundle adjustment is added, for the newly added points from PnP, before being saved into point cloud. Note that it is still extremely slow
bundle_adjustment = False

if os.path.exists("camera_matrix.npy"):
    ans = input("Refresh matrix(Y/N): ")
    if ans == "Y":
        K, _, _, _ = Find_Camera_Matrix()
        np.save("camera_matrix.npy", K)
    elif ans == "N":
        K = np.load("camera_matrix.npy")
    else:
        print("Error")
else:
    K, _, _, _ = Find_Camera_Matrix()
    np.save("camera_matrix.npy", K)

# Downscaling, if needed
# downscale = 2
# K[0,0] = K[0,0] / float(downscale)
# K[1,1] = K[1,1] / float(downscale)
# K[0,2] = K[0,2] / float(downscale)
# K[1,2] = K[1,2] / float(downscale)

# Camera pose matrix
pose0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
pose1 = np.empty((3, 4))
pose2 = np.empty((3, 4))

# Projection matrix
P1 = np.matmul(K, pose0)

# 3D point placeholder
points3d = np.zeros((1, 3))
points3d_ph = np.zeros((1, 3))
color = np.zeros((1, 3))
color_ph = np.zeros((1, 3))

images = sorted(os.listdir(img_dir))

# downscale = 2
# Setting the Reference two frames
# img0 = img_downscale(cv2.imread(img_dir + '/' + images[0]), downscale)
# img1 = img_downscale(cv2.imread(img_dir + '/' + images[1]), downscale)
img0 = cv2.imread(img_dir + '/' + images[0])
img1 = cv2.imread(img_dir + '/' + images[1])

pts0, pts1 = sift(img0, img1)
R, t, pts0, pts1 = Find_Essential_Matrix(pts0, pts1, K) # Includes RANSAC
flip = [[-1, 0, 0],[0, -1, 0],[0, 0, 1]]
# R = np.matmul(flip, R)
optical_flow_image = draw_optical_flow(img0, img1, pts0, pts1, output_path="optical_flow_0&1.png")
pose1[:3, :3] = np.matmul(R, pose0[:3, :3])
pose1[:3, 3] = pose0[:3, 3] + np.matmul(pose0[:3, :3], t.ravel())
P2 = np.matmul(K, pose1)

pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1)
# Check reprojection error
error, points_3d, _ = ReprojectionError(points_3d, pts1, pose1, K, homogenity = 1)
print("REPROJECTION ERROR1: ", error)
# Optimizing camera pose
Rot, trans, pts1, points_3d, pts0t = PnP(points_3d, pts1, K, np.zeros((5, 1), dtype=np.float32), pts0, initial=1)
# Refine and re-triangulate
Rt1 = np.hstack((Rot, trans))
P2 = np.matmul(K, Rt1)
pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1)
error, points_3d, _ = ReprojectionError(points_3d, pts1, pose1, K, homogenity = 1)
print("REPROJECTION ERROR2: ", error)

# Acquire new image to be added to the pipeline and acquire matches with image pair
# img2 = img_downscale(cv2.imread(img_dir + '/' + images[2]), downscale)
img2 = cv2.imread(img_dir + '/' + images[2])
pts_, pts2 = sift(img1, img2)
R, t, pts_, pts2 = Find_Essential_Matrix(pts_, pts2, K) # Includes RANSAC

pose2[:3, :3] = np.matmul(R, Rt1[:3, :3])
pose2[:3, 3] = Rt1[:3, 3] + np.matmul(Rt1[:3, :3], t.ravel())
P3 = np.matmul(K, pose2)

optical_flow_image = draw_optical_flow(img1, img2, pts_, pts2, output_path="optical_flow_1&2.png")

# Indexing
indx1, indx2, temp1, temp2 = common_points(pts1, pts_, pts2)
optical_flow_image = draw_optical_flow(img1, img2, temp1, temp2, output_path="optical_flow_1&2_noncommon.png")
com_pts2 = pts2[indx2]
com_pts_ = pts_[indx2]
com_pts1 = pts1[indx1]
optical_flow_image = draw_optical_flow(img0, img1, pts0[indx1], pts1[indx1], output_path="optical_flow_0&1_common.png")
# PnP with common points only
_, _, com_pts2, points_3d, com_pts_ = PnP(points_3d[indx1], com_pts2, K, np.zeros((5, 1), dtype=np.float32), com_pts_, initial = 0)

# Initiate point cloud with common points
points3d = np.vstack((points3d, points_3d))
pts1_reg = np.array(com_pts_, dtype=np.int32)
_colors = np.array([img1[l[1], l[0]] for l in pts1_reg])
colors = np.vstack((color, _colors))

# Triangulation with noncommon points
temp1, temp2, points2_3d = Triangulation(P2, P3, temp1, temp2)
# Projection error for third camera
error, _, _ = ReprojectionError(points2_3d, temp2, pose2, K, homogenity = 1)

print("REPROJECTION ERROR3: ", error)
# PnP noncommon points only
Rot, trans, temp2, points2_3d, temp1 = PnP(points2_3d, temp2, K, np.zeros((5, 1), dtype=np.float32), temp1, initial=1)
optical_flow_image = draw_optical_flow(img1, img2, temp1, temp2, output_path="optical_flow_1&2_afterP.png")
# Refine P3
Rt2 = np.hstack((Rot, trans))
P3 = np.matmul(K, Rt2)
points3dcopy = points3d.copy()
points3dcopy = (flip @ points3dcopy.T).T
view_point_cloud(points3dcopy, colors, pose0, Rt1, None)

# For testing
points23d = np.vstack((points3d, points2_3d))
pts1_reg = np.array(temp1, dtype=np.int32)
_colors_hold = np.array([img1[l[1], l[0]] for l in pts1_reg])
colors_hold = np.vstack((color, _colors_hold))
points23dcopy = points23d.copy()
points23dcopy = (flip @ points23dcopy.T).T
view_point_cloud(points23dcopy, colors_hold, None, Rt1, Rt2)

# Bundle Adjustment
points2_3d, temp2, Rt2 = BundleAdjustment(points2_3d, temp2, Rt2, K, 0.5) # gtol_thresh=0.5
P3 = np.matmul(K, Rt2)
optical_flow_image = draw_optical_flow(img1, img2, temp1, temp2, output_path="optical_flow_1&2_BA.png")
error, points2_3d, _ = ReprojectionError(points2_3d, temp2, Rt2, K, homogenity = 0)
print("REPROJECTION ERROR4: ", error)
# Placeholder, for points from BA, debugging purpose
points3d_ph = np.vstack((points3d_ph, points2_3d))
# Merge aligned points into main point cloud
points3d = np.vstack((points3d, points2_3d))
pts1_reg = np.array(temp2, dtype=np.int32)
_colors = np.array([img2[l[1], l[0]] for l in pts1_reg])
colors = np.vstack((colors, _colors))
color_ph = np.vstack((color_ph, _colors))

# plt.show()
cv2.destroyAllWindows()

# Point Cloud
points3dcopy = points3d.copy()
points3dcopy = (flip @ points3dcopy.T).T
points3dphcopy = points3d_ph.copy()
points3dphcopy = (flip @ points3dphcopy.T).T
view_point_cloud(points3dphcopy, color_ph, None, Rt1, Rt2)
view_point_cloud(points3dcopy, colors, pose0, Rt1, Rt2)
# Saving projection matrices for all the images
# np.savetxt('pose.csv', posearr, delimiter = '\n')