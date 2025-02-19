# Thee View Reconstruction with Bundle Adjustment
This repository uses only three images

## Disclaimer
Referenced from: [Multiview 3D Reconstruction Repository](https://github.com/Yashas120/Multiview-3D-Reconstruction/tree/main)<br>
Please use the above repository as reference for Multiview SfM

## Process
1. Camera matrix is found with checkerboard camera calibration. You may use 8+(10 is recommended) checkerboard images to find the matrix, or replace the npy file.
2. First and second image SIFT, RANSAC, triangulation, camera pose refinement, PnP etc
3. Second and third image SIFT, RANSAC, and then find common points
4. Camera pose refinement, PnP, triangulation
5. Use only common points, and add new points that are refined with Bundle Adjustment
