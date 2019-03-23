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
draw_geometries([pcd, mesh_frame])

np_points = np.asarray(pcd.points)
# pcd_new = PointCloud()
# pcd_new.points = Vector3dVector(xyz_points)
# draw_geometries([pcd_new])  # Visualize the point cloud

# In[]:
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(np_points)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# In[]:
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(np_points)

gmix = GaussianMixture(n_components=3)
y_gmix = gmix.fit_predict(np_points)

# In[]:
plt.scatter(np_points[y_gmix == 0, 0], np_points[y_gmix == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(np_points[y_gmix == 1, 0], np_points[y_gmix == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(np_points[y_gmix == 2, 0], np_points[y_gmix == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()

# In[]:
np_points = np.around(np_points, decimals=2)
pcd.points = Vector3dVector(np_points)
draw_geometries([pcd, mesh_frame])

# In[]:
x_range = np.max(np_points[:,0]) - np.min(np_points[:,0])
y_range = np.max(np_points[:,1]) - np.min(np_points[:,1])

