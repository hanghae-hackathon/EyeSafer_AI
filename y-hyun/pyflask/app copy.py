import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 비디오 파일 열기
cap = cv2.VideoCapture(r"C:\Users\joyon\EyeSafer_AI\y-hyun\testvideo\test.mp4")

# 3D 그래프 생성
fig = plt.figure(figsize=(20, 8)) 
ax = fig.add_subplot(111, projection='3d')

# 초기 프레임 읽기
ret, frame = cap.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
x, y = np.meshgrid(range(gray_frame.shape[1]), range(gray_frame.shape[0]))
surf = ax.plot_surface(x, y, gray_frame, cmap='viridis')


# z 레이블 및 축 제거
ax.set_zlabel('')
ax.set_zticks([])
ax.view_init(elev=-90, azim=-90)
# 그래프 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show(block=False)  # 창이 닫히지 않도록 하기 위해 block=False로 설정

while True: 
    # 비디오의 다음 프레임 읽기
    ret, frame = cap.read()

    # 프레임이 제대로 읽혔는지 확인
    if not ret:
        print("Failed to read video frame.")
        break

    # 프레임을 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 영상 데이터를 3D로 표현
    surf.remove()  # 이전 그래프 삭제
    surf = ax.plot_surface(x, y, gray_frame, cmap='viridis')
    # 창 업데이트
    plt.pause(0.01)

# 비디오 캡처 객체 해제
cap.release()

# 창 닫기
plt.close()


# fig = Figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_trisurf(frame_counts, red_areas, danger_levels, cmap='viridis')
#     ax.set_xlabel('Frame')
#     ax.set_ylabel('Red Box Area')
#     ax.set_zlabel('Danger Level')
#     ax.set_title('Video Frame in 3D')
#     output = io.BytesIO()
#     fig.savefig(output, format='png')
#     output.seek(0)
#     return Response(output.getvalue(), mimetype='image/png')