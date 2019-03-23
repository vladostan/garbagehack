from open3d import *
import numpy as np
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    
image_no = 6
thresh = 2

pcd = read_point_cloud('pcd/pcd/learn{}.pcd'.format(image_no))
np_points = np.asarray(pcd.points)
print(np_points.shape)
np_points = np_points[np.sqrt(np.power(np_points[:, 0], 2) + np.power(np_points[:, 1], 2)
                              + np.power(np_points[:, 2], 2)) < thresh, :]
print(np_points.shape)
np_points[:, 0] = np_points[:, 0] - np.mean(np_points[:, 0])
np_points[:, 1] = np_points[:, 1] - np.mean(np_points[:, 1])
np_points[:, 2] = np_points[:, 2] - np.mean(np_points[:, 2])
mesh_frame = create_mesh_coordinate_frame(size=0.6, origin=[0, 0, 0])

axis = [1, 0, 0]
theta = np.deg2rad(228)
rot_matrix = rotation_matrix(axis, theta)
print(rot_matrix)
result = np.matmul(rot_matrix, np_points.T).T
print(result.shape)
pcd.points = Vector3dVector(result)
#draw_geometries([pcd, mesh_frame])



# In[]:
np_points = np.asarray(pcd.points)
asd = np_points[np_points[:,-1]>0.01]
pcd.points = Vector3dVector(asd)
#draw_geometries([pcd, mesh_frame])

# In[]:
np_points = np.asarray(pcd.points)
qwe = np.zeros_like(asd)
qwe[:,:2] = np_points[:,:2]
pcd.points = Vector3dVector(qwe)
#draw_geometries([pcd, mesh_frame])

np_points = np.asarray(pcd.points)
# pcd_new = PointCloud()
# pcd_new.points = Vector3dVector(xyz_points)
# draw_geometries([pcd_new])  # Visualize the point cloud

# In[]:
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# In[]:
#kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
#y_kmeans = kmeans.fit_predict(np_points)
#
#gmix = GaussianMixture(n_components=3)
#y_gmix = gmix.fit_predict(np_points)
#
## In[]:
#plt.scatter(np_points[y_gmix == 0, 0], np_points[y_gmix == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
#plt.scatter(np_points[y_gmix == 1, 0], np_points[y_gmix == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
#plt.scatter(np_points[y_gmix == 2, 0], np_points[y_gmix == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
##plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
#plt.legend()
#plt.show()

# In[]:
decimals = 2
np_points = np.around(np_points, decimals=decimals)

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

np_points = unique_rows(np_points)
pcd.points = Vector3dVector(np_points)
#draw_geometries([pcd, mesh_frame])

# In[]:
import cv2

x_min = np.min(np_points[:,0])
y_min = np.min(np_points[:,1])
x_range = np.around(np.max(np_points[:,0]) - x_min, decimals=decimals)
y_range = np.around(np.max(np_points[:,1]) - y_min, decimals=decimals)

np_points_shifted = np_points[:,:2]
np_points_shifted[:,0] -= x_min
np_points_shifted[:,1] -= y_min

mask = np.zeros((int(x_range*math.pow(10,decimals)+1),int(y_range*math.pow(10,decimals)+1)), dtype=np.uint8)
mask = np.zeros((int(y_range*math.pow(10,decimals)+1),int(x_range*math.pow(10,decimals)+1)), dtype=np.uint8)

for (x,y) in np_points_shifted:
#    mask[int(x*math.pow(10,decimals)), int(y*math.pow(10,decimals))] = True
    mask[int(y*math.pow(10,decimals)), int(x*math.pow(10,decimals))] = True


plt.imshow(mask)

kernel = np.ones((2, 2), np.uint8)

closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

plt.imshow(closing)

# In[]:
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)

xy = centroids[1:].astype(np.int32)/math.pow(10,decimals)

# In[:
result2 = result.copy()

result2 = np.around(result2, decimals=decimals)

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

result2 = unique_rows(result2)

z = []
for j in range(len(xy)):
    for i in range(len(result2)):
        if result2[i,0] == np.around(xy[j,0]+x_min, decimals=decimals) and result2[i,1] == np.around(xy[j,1]+y_min, decimals=decimals):
            print(i)
            print(result2[i,2])
            z.append(result2[i,2])
        
xyz = np.zeros((len(xy), 3))
for i in range(len(xyz)):
    xyz[i] = [xy[i,0]+x_min, xy[i,1]+y_min, z[i]]

print(xyz)
 
colors = [[0, 1, 0] for i in range(len(pcd.points))]

pcd.points = Vector3dVector(result2)
pcd.colors = Vector3dVector(colors)
#draw_geometries([pcd, mesh_frame])

# In[]:
mesh_frame_a = create_mesh_coordinate_frame(size=0.1, origin=xyz[0])
mesh_frame_b = create_mesh_coordinate_frame(size=0.1, origin=xyz[1])
mesh_frame_c = create_mesh_coordinate_frame(size=0.1, origin=xyz[2])

draw_geometries([pcd, mesh_frame, mesh_frame_a, mesh_frame_b, mesh_frame_c])