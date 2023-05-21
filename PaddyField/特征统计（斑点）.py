import cv2
import numpy as np

# 读取灰度图像
image = cv2.imread('C:/image/image5.png', 0)

# 进行预处理，包括平滑滤波、边缘增强等操作

# 平滑滤波
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 边缘增强
edges = cv2.Canny(blurred, 50, 150)

# 进行统计分析
# 斑点大小
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
spot_sizes = [cv2.contourArea(contour) for contour in contours]

# 斑点密度
spot_density = len(contours)

# 斑点分布
spot_distribution = np.mean(spot_sizes)

# 打印统计结果
print("Spot Sizes:", spot_sizes)
print("Spot Density:", spot_density)
print("Spot Distribution:", spot_distribution)
