import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import entropy

# 读取图像
image = cv2.imread('C:/image/image5.png', 0)  # 以灰度模式读取图像

# 计算灰度共生矩阵
distances = [1]  # 距离
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 方向
glcm = greycomatrix(image, distances, angles, levels=256)

# 计算纹理特征
contrast = greycoprops(glcm, 'contrast')
energy = greycoprops(glcm, 'energy')

# 打印结果
print("Contrast:", contrast)#对比度
print("Energy:", energy)#能量

# 计算直方图
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# 归一化直方图
histogram /= histogram.sum()

# 计算熵
image_entropy = entropy(histogram, base=2)

# 打印结果（熵）
print("Image Entropy:", image_entropy)
