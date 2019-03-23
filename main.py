# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt

# In[]:
img_file = "inet"
img_file = "a (7)"
img_file = "crop"
img = plt.imread("imgs/" + img_file + ".jpg")
plt.imshow(img)

# In[]:
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')

# In[]:
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(hsv)

# In[]:
xyz = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
plt.imshow(xyz)

# In[]:
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
plt.imshow(lab)

# In[]:
luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
plt.imshow(luv)

# In[]:
light_brown = (250, 225, 200)
dark_brown = (50, 25, 0)
#dark_brown = (33,14,0)

# In[]:
lb_square = np.full((10, 10, 3), light_brown, dtype=np.uint8) / 255.0
db_square = np.full((10, 10, 3), dark_brown, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(lb_square)
plt.subplot(1, 2, 2)
plt.imshow(db_square)
plt.show()

# In[]:
mask = ~cv2.inRange(img, dark_brown, light_brown)
result = cv2.bitwise_and(img, img, mask=mask)

plt.subplot(1, 2, 1)
plt.imshow(mask, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

# In[]:
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

#areas = np.sort(stats[:,-1])[::-1][1:]
sizes = stats[1:, -1] 
nb_components -= 1

min_size = 250

mask2 = np.zeros((output.shape))

for i in range(0, nb_components):
    if sizes[i] >= min_size:
        mask2[output == i + 1] = 255

plt.imshow(mask2)

# In[]:
kernel = np.ones((5,5),np.uint8)

dilation = cv2.dilate(mask2, kernel, iterations = 5)
closing = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

plt.imshow(dilation)

# In[]:
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilation.astype(np.uint8), connectivity=8)

xy = centroids[1:].astype(np.int32)

# In[]:
img_copy = img.copy()

for i in range(len(xy)):
    cv2.circle(img_copy, tuple(xy[i]), 15, (255,0,255), -1)

plt.imshow(img_copy)