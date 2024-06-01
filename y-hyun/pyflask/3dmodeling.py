import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 2D 영상 데이터 생성 (예: 10x10 픽셀 영상)
image = np.random.rand(10, 10)

# 2D 그래프 생성
x, y = np.meshgrid(range(image.shape[0]), range(image.shape[1]))

# 3D 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 영상 데이터를 3D로 표현
ax.plot_surface(x, y, image, cmap='viridis')

# 그래프 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 그래프 표시
plt.show()
