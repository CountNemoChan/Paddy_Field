import cv2
import numpy as np
import matplotlib.pyplot as plt

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



# 定义网格大小
width = image.shape[1]
height = image.shape[0]

# 定义网格步长
dx = 1.0  # x方向步长
dy = 1.0  # y方向步长

# 定义时间步长和总迭代次数
dt = 0.1  # 时间步长
iterations = 1000  # 总迭代次数

# 定义初始条件
initial_condition = np.zeros((height, width), dtype=np.float32)
initial_condition[spot_image[:,:,0] > 0] = 1.0

# 初始化解数组
solution = np.copy(initial_condition)

# 定义扩散系数(具体值由coef程序确定）
Dx = 1.0
Dy = 1.0

# 初始化解数组
solution = np.copy(initial_condition)

# 执行迭代求解
for _ in range(iterations):
    # 计算下一个时间步的解
    next_solution = np.copy(solution)

    # 在网格内部进行离散化的反应扩散计算
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            next_solution[i, j] = solution[i, j] + dt * (
                    Dx * (solution[i + 1, j] + solution[i - 1, j] - 2 * solution[i, j]) / dx ** 2 +
                    Dy * (solution[i, j + 1] + solution[i, j - 1] - 2 * solution[i, j]) / dy ** 2)

    # 更新解数组
    solution = np.copy(next_solution)
    
    
print(solution)
# 可视化数值解结果
plt.imshow(solution, cmap='hot', origin='lower')#输出热度图，使数值解可视化
plt.colorbar()
plt.show()

