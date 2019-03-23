# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt

# In[]:
img_file = "learn44"
img = plt.imread("rgbd/disparity/" + img_file + ".png")
plt.imshow(img)

# In[]:
from PIL import Image

focalLength = 938.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000

rgb_file = "rgbd/image_color/" + img_file + ".png"
depth_file = "rgbd/disparity/" + img_file + ".png"

def generate_pointcloud(rgb_file,depth_file,ply_file):

    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file).convert('I')

    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")

    points = []    
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            Z = depth.getpixel((u,v)) / scalingFactor
            print(Z)
            if Z==0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
            
    file = open(ply_file, "w")
    file.write('''ply
               format ascii 1.0
               element vertex %d
               property float x
               property float y
               property float z
               property uchar red
               property uchar green
               property uchar blue
               property uchar alpha
               end_header
               %s
               '''%(len(points),"".join(points)))
    file.close()
            
ply_file = "asd.ply"
generate_pointcloud(rgb_file, depth_file, ply_file)