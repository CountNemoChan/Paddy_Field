import cv2
import numpy as np

# 读取图像
image = cv2.imread('C:/image/image5.png', 0)  # 以灰度模式读取图像

# 使用Sobel算子进行边缘检测
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edges = cv2.sqrt(cv2.addWeighted(sobelx**2, 0.5, sobely**2, 0.5, 0))

# 形态学操作优化边缘
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# 显示结果
cv2.imshow("Original Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Optimized Edges", eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
