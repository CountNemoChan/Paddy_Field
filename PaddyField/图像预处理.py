#图像预处理
import cv2

# 读取PNG图像文件

image = cv2.imread('C:/image/image5.png')

# 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 平滑滤波
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 边缘增强
edges = cv2.Canny(blurred, 50, 150)

# 显示预处理结果
cv2.imshow("Original Image", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Blurred", blurred)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()