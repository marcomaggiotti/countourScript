import cv2
import os

scr1_file_path = '../../imgs/8918_00007.tif'
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
cv2.imwrite('../../data/mask_bitwise/' + baseName +'_1.jpg', dst1)
cv2.imwrite('../../data/mask_bitwise/' + baseName +'_0.jpg', dst0)

# True
# ![](data/dst/opencv_bitwise_and.jpg)