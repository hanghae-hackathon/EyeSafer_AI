import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 비디오 파일 열기
cap = cv2.VideoCapture(r"C:\Users\joyon\EyeSafer_AI\y-hyun\testvideo\test6.mp4")

while True: 
    # 비디오의 첫 번째 프레임 읽기
    ret, frame = cap.read()

    # 프레임이 제대로 읽혔는지 확인
    if not ret:
        print("Failed to read video frame.")
        exit()

    # 프레임을 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2D 그래프 생성
    x, y = np.meshgrid(range(gray_frame.shape[1]), range(gray_frame.shape[0]))

    # 3D 그래프 생성
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 영상 데이터를 3D로 표현
    ax.plot_surface(x, y, gray_frame, cmap='viridis')

    # 그래프 레이블 및 타이틀 설정
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Video Frame in 3D')

    # 그래프 표시
    plt.show()

    # 비디오 캡처 객체 해제
    cap.release()
