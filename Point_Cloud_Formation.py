import numpy as np
import open3d as o3d

def view_point_cloud(point_cloud, colors, Rt0=None, Rt1=None, Rt2=None):
    out_points = point_cloud.reshape(-1, 3) 
    out_colors = colors.reshape(-1, 3) / 255

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_points)
    pcd.colors = o3d.utility.Vector3dVector(out_colors)

    # Visualization objects
    vis_objects = [pcd]

    # Add camera frustums if provided
    if Rt0 is not None:
        vis_objects.append(create_camera_frustum(Rt0, scale=0.1, color=[1, 0, 0]))  # Red
    if Rt1 is not None:
        vis_objects.append(create_camera_frustum(Rt1, scale=0.1, color=[0, 1, 0]))  # Green
    if Rt2 is not None:
        vis_objects.append(create_camera_frustum(Rt2, scale=0.1, color=[0, 0, 1]))  # Blue


    # Visualize with cameras
    o3d.visualization.draw_geometries(vis_objects, window_name="Point Cloud Viewer",
                                    width=800, height=600, left=50, top=50,
                                    point_show_normal=False)

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
    return np.array(indx1), np.array(indx2), temp_array1, temp_array2

def create_camera_frustum(Rt, scale=10, color=[1, 0, 0], arrow_scale=5):
    R = Rt[:, :3]  # Extract Rotation (3x3)
    t = Rt[:, 3]   # Extract Translation (3x1)

    # Define frustum points in camera coordinate system
    frustum_points = np.array([
        [0, 0, 0],  # Camera center
        [1, 1, 1], [1, -1, 1],
        [-1, -1, 1], [-1, 1, 1]
    ]) * scale
    
    # Define arrow tip for upward direction
    arrow_tip = np.array([0, 0.1, 0]) * arrow_scale
    
    # Convert to world coordinates
    frustum_points = (R @ frustum_points.T).T + t.reshape((1, 3))
    arrow_tip = (R @ arrow_tip.T).T + t.reshape((1, 3))

    # Append arrow tip to points
    frustum_points = np.vstack([frustum_points, arrow_tip])
    arrow_tip_idx = len(frustum_points) - 1  # Index of the arrow tip

    # Define frustum edges
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Camera center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Connect corners
        [0, arrow_tip_idx]  # Arrow line
    ]
    
    # Define colors
    colors = [color] * len(lines)  # Ensure all lines, including the arrow, have the same color

    # Create Open3D LineSet
    camera_frustum = o3d.geometry.LineSet()
    camera_frustum.points = o3d.utility.Vector3dVector(frustum_points)
    camera_frustum.lines = o3d.utility.Vector2iVector(lines)
    camera_frustum.colors = o3d.utility.Vector3dVector(colors)

    return camera_frustum
