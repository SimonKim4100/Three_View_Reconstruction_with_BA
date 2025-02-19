import numpy as np
import open3d as o3d

def view_point_cloud(point_cloud, colors):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3) / 255
    print("shape: ", out_colors.shape, out_points.shape)
    # verts = np.hstack([out_points, out_colors])

    # # Point Cloud Cleaner
    # mean = np.mean(out_points, axis=0)
    # temp = out_points - mean
    # dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    # # Remove outliers
    # indx = np.where(dist < np.mean(dist) + 1000000)
    # out_points = out_points[indx]
    # out_colors = out_colors[indx]
    # print("shape_in: ", out_colors.shape, out_points.shape)

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_points)
    pcd.colors = o3d.utility.Vector3dVector(out_colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Viewer",
                                      width=800, height=600, left=50, top=50)

def common_points(pts1, pts2, pts3):
    indx1 = []
    indx2 = []
    for i in range(pts1.shape[0]):
        a = np.where(pts2 == pts1[i, :])
        if a[0].size == 0:
            pass
        else:
            indx1.append(i)
            indx2.append(a[0][0])

    temp_array1 = np.ma.array(pts2, mask=False)
    temp_array1.mask[indx2] = True
    temp_array1 = temp_array1.compressed()
    temp_array1 = temp_array1.reshape(int(temp_array1.shape[0] / 2), 2)

    temp_array2 = np.ma.array(pts3, mask=False)
    temp_array2.mask[indx2] = True
    temp_array2 = temp_array2.compressed()
    temp_array2 = temp_array2.reshape(int(temp_array2.shape[0] / 2), 2)
    print("Shape New Array", temp_array1.shape, temp_array2.shape)
    return np.array(indx1), np.array(indx2), temp_array1, temp_array2
