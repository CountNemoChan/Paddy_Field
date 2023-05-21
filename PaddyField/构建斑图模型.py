import cv2
import numpy as np

# 读取输入图片
image = cv2.imread('C:/image/image5.png')

# 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 平滑滤波
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 边缘增强
edges = cv2.Canny(blurred, 50, 150)

# 使用阈值分割方法检测斑点
_, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

# 连通区域分析，获取斑点轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制斑点轮廓
spot_image = np.zeros_like(image)  # 创建与原图大小相同的空白斑图模型
cv2.drawContours(spot_image, contours, -1, (255, 255, 255), 2)

# 显示原图、预处理结果和斑图模型
cv2.imshow("Original Image", image)
cv2.imshow("Preprocessed Image", edges)
cv2.imshow("Spot Image", spot_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
