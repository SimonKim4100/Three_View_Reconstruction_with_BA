# Structure from Motion
# Authors: Arihant Gaur and Saurabh Kemekar
# Organization: IvLabs, VNIT


import cv2
import numpy as np
import os
from scipy.optimize import least_squares
import copy
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt

from Features import img_downscale, find_features
from Estimate_Essential_Matrix import find_Essential_Matrix
from Point_Triangulation import Triangulation
from Perspective_n_Point import PnP
from Reprojection_Error import ReprojectionError
from Bundle_Adjustment import BundleAdjustment
from Point_Cloud_Formation import camera_orientation, to_ply, common_points

# Current Path Directory
path = os.getcwd()

# Input the directory where the images are kept. Note that the images have to be named in order for this particular implementation
#img_dir = path + '/Sample Dataset/'
# img_dir = r"C:\Users\sachi\Downloads\photos\images"
# img_dir = r"C:\Users\sachi\Downloads\GustavIIAdolf"
# img_dir = r"C:\Users\sachi\Downloads\Hornbake"
# img_dir = r"C:\Users\sachi\Downloads\Testudo"
img_dir = r"C:\Users\sachi\Downloads\Data"
# A provision for bundle adjustment is added, for the newly added points from PnP, before being saved into point cloud. Note that it is still extremely slow
bundle_adjustment = False

# Input Camera Intrinsic Parameters
K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])

# Suppose if computationally heavy, then the images can be downsampled once. Note that downsampling is done in powers of two, that is, 1,2,4,8,...
downscale = 2
K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)


posearr = K.ravel()
R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
R_t_1 = np.empty((3, 4))

P1 = np.matmul(K, R_t_0)
Pref = P1
P2 = np.empty((3, 4))

Xtot = np.zeros((1, 3))
colorstot = np.zeros((1, 3))


img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
    if '.jpg' in img.lower() or '.png' in img.lower():
        images = images + [img]
i = 0
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()



densify = False  # Added in case we will merge densification step in this code. Mostly it will be considered separately, though still added here.

downscale = 2
# Setting the Reference two frames
img0 = img_downscale(cv2.imread(img_dir + '/' + images[i]), downscale)
img1 = img_downscale(cv2.imread(img_dir + '/' + images[i + 1]), downscale)

# plt.imshow(img0)
# plt.imshow(img1)

pts0, pts1 = find_features(img0, img1)

R, t, pts0, pts1 = find_Essential_Matrix(pts0, pts1, K)

# print(mask.shape)
# print('shape2', pts0.shape, pts1.shape)
R_t_1[:3, :3] = np.matmul(R, R_t_0[:3, :3])
R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3, :3], t.ravel())

P2 = np.matmul(K, R_t_1)

# Triangulation is done for the first image pair. The poses will be set as reference, that will be used for increemental SfM
pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1, K, repeat=False)
# Backtracking the 3D points onto the image and calculating the reprojection error. Ideally it should be less than one.
# If found to be the culprit for an incorrect point cloud, enable Bundle Adjustment
error, points_3d, repro_pts = ReprojectionError(points_3d, pts1, R_t_1, K, homogenity = 1)
print("REPROJECTION ERROR: ", error)
Rot, trans, pts1, points_3d, pts0t = PnP(points_3d, pts1, K, np.zeros((5, 1), dtype=np.float32), pts0, initial=1)
#Xtot = np.vstack((Xtot, points_3d))

R = np.eye(3)
t = np.array([[0], [0], [0]], dtype=np.float32)

# Here, the total images to be taken into consideration can be varied. Ideally, the whole set can be used, or a part of it. For whole lot: use tot_imgs = len(images) - 2
tot_imgs = len(images) - 2 

posearr = np.hstack((posearr, P1.ravel()))
posearr = np.hstack((posearr, P2.ravel()))

gtol_thresh = 0.5
#camera_orientation(path, mesh, R_t_0, 0)
#camera_orientation(path, mesh, R_t_1, 1)

for i in tqdm(range(tot_imgs)):
    # Acquire new image to be added to the pipeline and acquire matches with image pair
    img2 = img_downscale(cv2.imread(img_dir + '/' + images[i + 2]), downscale)

    # pts0,pts1 = find_features(img1,img2)

    pts_, pts2 = find_features(img1, img2)
    if i != 0:
        pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1, K, repeat = False)
        pts1 = pts1.T
        points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
        points_3d = points_3d[:, 0, :]
    
    # There gone be some common point in pts1 and pts_
    # we need to find the indx1 of pts1 match with indx2 in pts_
    indx1, indx2, temp1, temp2 = common_points(pts1, pts_, pts2)
    com_pts2 = pts2[indx2]
    com_pts_ = pts_[indx2]
    com_pts0 = pts0.T[indx1]
    # We have the 3D - 2D Correspondence for new image as well as point cloud obtained from before. The common points can be used to find the world coordinates of the new image
    # using Perspective - n - Point (PnP)
    Rot, trans, com_pts2, points_3d, com_pts_ = PnP(points_3d[indx1], com_pts2, K, np.zeros((5, 1), dtype=np.float32), com_pts_, initial = 0)
    # Find the equivalent projection matrix for new image
    Rtnew = np.hstack((Rot, trans))
    Pnew = np.matmul(K, Rtnew)

    #print(Rtnew)
    error, points_3d, _ = ReprojectionError(points_3d, com_pts2, Rtnew, K, homogenity = 0)
   
    temp1, temp2, points_3d = Triangulation(P2, Pnew, temp1, temp2, K, repeat = False)
    error, points_3d, _ = ReprojectionError(points_3d, temp2, Rtnew, K, homogenity = 1)
    print("Reprojection Error: ", error)
    # We are storing the pose for each image. This will be very useful during multiview stereo as this should be known
    posearr = np.hstack((posearr, Pnew.ravel()))

    # If bundle adjustment is considered. gtol_thresh represents the gradient threshold or the min jump in update that can happen. If the jump is smaller, optimization is terminated.
    # Note that most of the time, the pipeline yield a reprojection error less than 0.1! However, it is often too slow, often close to half a minute per frame!
    # For point cloud registration, the points are updated in a NumPy array. To visualize the object, the corresponding BGR color is also updated, which will be merged
    # at the end with the 3D points
    if bundle_adjustment:
        print("Bundle Adjustment...")
        points_3d, temp2, Rtnew = BundleAdjustment(points_3d, temp2, Rtnew, K, gtol_thresh)
        Pnew = np.matmul(K, Rtnew)
        error, points_3d, _ = ReprojectionError(points_3d, temp2, Rtnew, K, homogenity = 0)
        print("Minimized error: ",error)
        Xtot = np.vstack((Xtot, points_3d))
        pts1_reg = np.array(temp2, dtype=np.int32)
        colors = np.array([img2[l[1], l[0]] for l in pts1_reg])
        colorstot = np.vstack((colorstot, colors))
    else:
        Xtot = np.vstack((Xtot, points_3d[:, 0, :]))
        pts1_reg = np.array(temp2, dtype=np.int32)
        colors = np.array([img2[l[1], l[0]] for l in pts1_reg.T])
        colorstot = np.vstack((colorstot, colors)) 
    #camera_orientation(path, mesh, Rtnew, i + 2)    


    R_t_0 = np.copy(R_t_1)
    P1 = np.copy(P2)
    plt.scatter(i, error)
    plt.pause(0.05)

    img0 = np.copy(img1)
    img1 = np.copy(img2)
    pts0 = np.copy(pts_)
    pts1 = np.copy(pts2)
    #P1 = np.copy(P2)
    P2 = np.copy(Pnew)
    # cv2.imshow('image', img2)
    # if cv2.waitKey(1) & 0xff == ord('q'):
    #     break

# plt.show()
cv2.destroyAllWindows()

# Finally, the obtained points cloud is registered and saved using open3d. It is saved in .ply form, which can be viewed using meshlab
print("Processing Point Cloud...")
print(Xtot.shape, colorstot.shape)
to_ply(path, Xtot, colorstot, densify)
print("Done!")
# Saving projection matrices for all the images
np.savetxt('pose.csv', posearr, delimiter = '\n')

