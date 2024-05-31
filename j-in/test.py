import torch
import cv2
import numpy as np

# 모델 로드 (사전 훈련된 YOLOv5 모델 사용)
model_path = '/Users/ijaein/Desktop/EyeSafer_AI/yolov5/runs/train/exp3/weights/best.pt'  # 올바른 경로를 지정하세요
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# 경고 인원 수 설정
alert_threshold = 3

# 비디오 파일 경로 설정
video_path = '/Users/ijaein/Desktop/EyeSafer_AI/KakaoTalk_Video_2024-06-01-02-43-18.mp4'  # 여기에 비디오 파일 경로를 입력하세요

# 비디오 캡처 시작
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 객체 탐지
    results = model(frame)

    # 탐지된 사람 수 계산 및 위치 저장
    persons = [x for x in results.xyxy[0] if int(x[-1]) == 0]  # class 0은 사람을 의미
    num_persons = len(persons)

    # 탐지된 인원 수 화면에 출력
    cv2.putText(frame, f'Total Persons: {num_persons}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 사람들 간의 거리 계산
    close_persons = []
    for i in range(num_persons):
        for j in range(i + 1, num_persons):
            box1 = persons[i][:4]
            box2 = persons[j][:4]
            distance = calculate_distance(box1, box2)
            if distance < 100:  # 가까운 거리의 기준 (픽셀 단위로 조정 가능)
                close_persons.append((box1, box2))

    # 특정 구역에 모여 있는 사람들의 중심 계산
    if len(close_persons) >= alert_threshold:
        centers = []
        for box1, box2 in close_persons:
            centers.append(get_center(box1))
            centers.append(get_center(box2))

        # 중심 좌표의 평균 계산
        centers = np.array(centers)
        avg_center = np.mean(centers, axis=0)

        # 경고 영역 설정
        x_min = int(np.min(centers[:, 0]) - 50)
        y_min = int(np.min(centers[:, 1]) - 50)
        x_max = int(np.max(centers[:, 0]) + 50)
        y_max = int(np.max(centers[:, 1]) + 50)

        # 투명한 빨간 박스 그리기
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
        alpha = 0.4  # 투명도
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 결과 표시
    for *xyxy, conf, cls in persons:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()