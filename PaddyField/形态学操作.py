import cv2

# 读取图像
image = cv2.imread('C:/image/image5.png', 0)  # 以灰度模式读取图像

# 进行腐蚀操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
erosion = cv2.erode(image, kernel, iterations=1)

# 进行膨胀操作
dilation = cv2.dilate(image, kernel, iterations=1)


# 显示结果
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
