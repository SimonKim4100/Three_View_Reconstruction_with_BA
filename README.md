# Thee View Reconstruction with Bundle Adjustment
This repository uses only three images

## Process
1. Camera matrix is found with [checkerboard camera calibration](https://github.com/SimonKim4100/Camera_Calibration_and_Visualization). You may use 8+(10 is recommended) checkerboard images to find the matrix, or replace the npy file.
2. First and second image SIFT, RANSAC, triangulation, camera pose refinement, PnP etc
3. Second and third image SIFT, RANSAC, and then find common points
4. Camera pose refinement, PnP, triangulation
5. Use only common points, and add new points with third image that are refined with Bundle Adjustment

## Note
This repository does not use the original method implemented by COLMAP. Thus, you may get a error spike if you use your own images. Try adjusting camera angles between images if you get errors.
