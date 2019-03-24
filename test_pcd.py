from open3d import *
import numpy as np
import math
import scipy.cluster.hierarchy as hcluster
import random


def generate_colors(num):
    def r(): return random.randint(0, 255)
    return [[r()/255, r()/255, r()/255] for _ in range(num)]


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


def get_trash_coords(image_no=10):
    thresh = 2
    pcd = read_point_cloud('pcd/pcd/learn{}.pcd'.format(image_no))
    np_points = np.asarray(pcd.points)
    print(np_points.shape)
    np_points = np_points[np.sqrt(np.power(np_points[:, 0], 2) + np.power(np_points[:, 1], 2)
                                  + np.power(np_points[:, 2], 2)) < thresh, :]
    print(np_points.shape)

    x_mean, y_mean, z_mean = np.mean(np_points[:, 0]), np.mean(
        np_points[:, 1]), np.mean(np_points[:, 2])
    np_points[:, 0] = np_points[:, 0] - x_mean
    np_points[:, 1] = np_points[:, 1] - y_mean
    np_points[:, 2] = np_points[:, 2] - z_mean

    axis = [1, 0, 0]
    theta_deg = 228
    theta = np.deg2rad(theta_deg)
    rot_matrix = rotation_matrix(axis, theta)
    print(rot_matrix)
    result = np.matmul(rot_matrix, np_points.T).T
    print(result.shape)
    result = result[result[:, 1] < 0.45, :]
    pcd_new = PointCloud()
    pcd_new.points = Vector3dVector(result[result[:, 2] <= 0.01, :])
    X = result[result[:, 2] > 0.01, :]

    idxes = np.random.choice(X.shape[0], int(X.shape[0]*0.1))
    X = X[idxes]

    cluster_peaks = []
    thresh = 0.02
    clusters = hcluster.fclusterdata(X, thresh, criterion="distance")
    for i in np.unique(clusters):
        curr_cluster = X[clusters == i, :]
        cluster_peaks.append(
            curr_cluster[np.argmax(curr_cluster[:, 2], axis=0), :])

    num_clusters = np.unique(clusters).shape[0]
    clrs = generate_colors(num_clusters)
    print(num_clusters)
    colors = np.array([clrs[x-1] for x in clusters])
    trash_coordinates = cluster_peaks
    pcd.points = Vector3dVector(X)
    pcd.colors = Vector3dVector(colors)

    viz_objects = []
    for sp in trash_coordinates:
        print(sp)
        mesh_sphere = create_mesh_sphere(radius=0.01)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0, 0, 1])
        your_transform = np.asarray(
            [[1, 0, 0, sp[0]],
             [0, 1, 0, sp[1]],
             [0, 0, 1, sp[2]],
             [0.0, 0.0, 0.0, 1.0]])
        mesh_sphere.transform(your_transform)
        viz_objects.append(mesh_sphere)

    mesh_frame = create_mesh_coordinate_frame(
        size=0.6, origin=[x_mean, y_mean, z_mean])
    viz_objects.append(pcd)
    viz_objects.append(mesh_frame)
    viz_objects.append(pcd_new)
    draw_geometries(viz_objects)

    return trash_coordinates


for i in range(44):
    get_trash_coords(i)
