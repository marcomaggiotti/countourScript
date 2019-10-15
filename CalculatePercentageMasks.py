import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

annotatedColor = 0
scr1_file_path = '../../imgs/8918_00007.tif'

img = cv2.imread(scr1_file_path)
plt.imshow(img)
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

print(len(flags))
print(flags[40])

baseNameWithExtension = os.path.basename(scr1_file_path)
baseName = os.path.splitext(baseNameWithExtension)[0]
print(" The basename is " , baseName)
src1 = cv2.imread(scr1_file_path)

scr2_file_path = '../../masks/8918_00007_mask.png'
src2 = cv2.imread(scr2_file_path)

src2 = cv2.resize(src2, src1.shape[1::-1])

print(src2.shape)
# (225, 400, 3)

print(src2.dtype)
# uint8

dst1 = cv2.bitwise_and(src1, src2)
dst0 = cv2.bitwise_and(src1, (255 - src2))

plt.imshow(dst1)
plt.show()

plt.imshow(dst0)
plt.show()

n_white_pix = np.sum(dst1 == annotatedColor)
print(n_white_pix)

n_white_pix = np.sum(dst0 == annotatedColor)
print(n_white_pix)