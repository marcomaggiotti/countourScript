import cv2
import os

image_file_path = '../../imgs/8918_00007.tif'
black_color_annotation = 255
dir_mask_bitwise = '../../data/mask_bitwise/'

baseNameWithExtension = os.path.basename(image_file_path)
baseName = os.path.splitext(baseNameWithExtension)[0]
print(" The basename is " , baseName)

src1 = cv2.imread(image_file_path)

mask_file_path = '../../masks/8918_00007_mask.png'
src2 = cv2.imread(mask_file_path)

src2 = cv2.resize(src2, src1.shape[1::-1])

print(src2.shape)
# (225, 400, 3)

print(src2.dtype)
# uint8


dst1 = cv2.bitwise_and(src1, src2)
dst0 = cv2.bitwise_and(src1, ( black_color_annotation - src2))

cv2.imwrite(dir_mask_bitwise + baseName +'_1.jpg', dst1)
cv2.imwrite(dir_mask_bitwise + baseName +'_0.jpg', dst0)

# True
# ![](data/dst/opencv_bitwise_and.jpg)