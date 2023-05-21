import cv2

# 读取图像
image = cv2.imread('C:/image/image5.png')

# 将图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用阈值分割
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 显示结果
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
