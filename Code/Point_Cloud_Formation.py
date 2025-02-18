import numpy as np
import cv2
import open3d as o3d
import copy

# Camera pose registration. This is quite useful in visualizing the pose for each camera, along with the point cloud. It is currently disabled. Will be fixed later.
def camera_orientation(path, mesh, R_T, i):
    T = np.zeros((4, 4))
    T[:3, ] = R_T
    T[3, :] = np.array([0, 0, 0, 1])
    new_mesh = copy.deepcopy(mesh).transform(T)
    # print(new_mesh)
    #new_mesh.scale(0.5, center=new_mesh.get_center())
    o3d.io.write_triangle_mesh(path + "/Point_Cloud/camerapose" + str(i) + '.ply', new_mesh)
    return

def to_ply(path, point_cloud, colors, densify):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(out_colors.shape, out_points.shape)
    verts = np.hstack([out_points, out_colors])

    # cleaning point cloud
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    #print(dist.shape, np.mean(dist))
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    #print( verts.shape)
    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
    if not densify:
        with open(path + '/Point_Cloud/sparse.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
    else:
        with open(path + '/Point_Cloud/dense.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
            



def common_points(pts1, pts2, pts3):
    '''Here pts1 represent the points image 2 find during 1-2 matching
    and pts2 is the points in image 2 find during matching of 2-3 '''
    indx1 = []
    indx2 = []
    for i in range(pts1.shape[0]):
        a = np.where(pts2 == pts1[i, :])
        if a[0].size == 0:
            pass
        else:
            indx1.append(i)
            indx2.append(a[0][0])

    '''temp_array1 and temp_array2 will which are not common '''
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
