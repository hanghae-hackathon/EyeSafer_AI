# import torch
# import cv2
# import numpy as np

# # 모델 로드 (사전 훈련된 YOLOv5 모델 사용)
# model_path = '/Users/ijaein/Desktop/ll/j-in/EyeSafer_AI/yolov5/runs/train/exp3/weights/best.pt'  # 올바른 경로를 지정하세요
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# # 경고 인원 수 설정
# alert_threshold = 5  # 5명 이상일 경우 표시
# confidence_threshold = 0.3  # 인식 정확도 0.3 이상인 경우만 사람으로 인식

# # 비디오 파일 경로 설정
# video_path = '/Users/ijaein/Desktop/ll/j-in/EyeSafer_AI/video/화면 기록 2024-06-01 오전 5.35.39.mov'  # 여기에 비디오 파일 경로를 입력하세요

# # 비디오 캡처 시작
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# def calculate_iou(box1, box2):
#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2

#     xi1 = max(x1, x3)
#     yi1 = max(y1, y3)
#     xi2 = min(x2, x4)
#     yi2 = min(y2, y4)

#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (x4 - x3) * (y4 - y3)

#     union_area = box1_area + box2_area - inter_area
#     iou = inter_area / union_area

#     return iou

# def get_center(box):
#     return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# def find_clusters(persons, iou_threshold):
#     clusters = []
#     visited = set()

#     for i, person in enumerate(persons):
#         if i in visited:
#             continue

#         cluster = [person]
#         visited.add(i)
#         stack = [i]

#         while stack:
#             current = stack.pop()
#             for j, other in enumerate(persons):
#                 if j not in visited and calculate_iou(persons[current][:4], other[:4]) > iou_threshold:
#                     visited.add(j)
#                     stack.append(j)
#                     cluster.append(other)

#         clusters.append(cluster)
#     return clusters

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # 객체 탐지
#     results = model(frame)

#     # 탐지된 사람 수 계산 및 위치 저장
#     persons = [x for x in results.xyxy[0] if int(x[-1]) == 0 and x[4] >= confidence_threshold]  # class 0은 사람을 의미, 인식 정확도 0.3 이상
#     num_persons = len(persons)

#     # 탐지된 인원 수 화면에 출력
#     cv2.putText(frame, f'Total Persons: {num_persons}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#     # 사람들 간의 거리 계산 및 클러스터링
#     clusters = find_clusters(persons, 0.5)  # IoU가 0.5 이상인 경우 클러스터로 간주

#     for cluster in clusters:
#         if len(cluster) >= alert_threshold:
#             # 클러스터의 중심 좌표 계산
#             centers = [get_center(person[:4]) for person in cluster]
#             centers = np.array(centers)
#             avg_center = np.mean(centers, axis=0)

#             # 경고 영역 설정
#             x_min = int(np.min(centers[:, 0]) - 20)
#             y_min = int(np.min(centers[:, 1]) - 20)
#             x_max = int(np.max(centers[:, 0]) + 20)
#             y_max = int(np.max(centers[:, 1]) + 20)

#             # 투명한 빨간 박스 그리기
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
#             alpha = 0.4  # 투명도
#             frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

#             # 경고 메시지 출력
#             cv2.putText(frame, 'DANGER: Crowding Detected!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # 결과 표시
#     for *xyxy, conf, cls in persons:
#         label = f'{model.names[int(cls)]} {conf:.2f}'
#         cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
#         cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     cv2.imshow('Frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # import torch
# # import cv2
# # import numpy as np
# # import yt_dlp as youtube_dl

# # # 모델 로드 (사전 훈련된 YOLOv5 모델 사용)
# # model_path = '/Users/ijaein/Desktop/ll/j-in/EyeSafer_AI/yolov5/runs/train/exp3/weights/best.pt'  # 올바른 경로를 지정하세요
# # model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# # # 경고 인원 수 설정
# # alert_threshold = 5  # 5명 이상일 경우 표시
# # confidence_threshold = 0.3  # 인식 정확도 0.3 이상인 경우만 사람으로 인식

# # # YouTube 라이브 스트리밍 URL 설정
# # url = 'https://www.youtube.com/watch?v=gFRtAAmiFbE'

# # # 비디오 스트리밍 URL 가져오기
# # def get_video_url(url):
# #     ydl_opts = {
# #         'format': 'best',
# #         'noplaylist': True,
# #     }
# #     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
# #         info_dict = ydl.extract_info(url, download=False)
# #         video_url = info_dict.get('url', None)
# #     return video_url

# # video_url = get_video_url(url)

# # # 비디오 캡처 시작
# # cap = cv2.VideoCapture(video_url)

# # if not cap.isOpened():
# #     print("Error: Could not open video.")
# #     exit()

# # def calculate_iou(box1, box2):
# #     x1, y1, x2, y2 = box1
# #     x3, y3, x4, y4 = box2

# #     xi1 = max(x1, x3)
# #     yi1 = max(y1, y3)
# #     xi2 = min(x2, x4)
# #     yi2 = min(y2, y4)

# #     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
# #     box1_area = (x2 - x1) * (y2 - y1)
# #     box2_area = (x4 - x3) * (y4 - y3)

# #     union_area = box1_area + box2_area - inter_area
# #     iou = inter_area / union_area

# #     return iou

# # def get_center(box):
# #     return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# # def find_clusters(persons, iou_threshold):
# #     clusters = []
# #     visited = set()

# #     for i, person in enumerate(persons):
# #         if i in visited:
# #             continue

# #         cluster = [person]
# #         visited.add(i)
# #         stack = [i]

# #         while stack:
# #             current = stack.pop()
# #             for j, other in enumerate(persons):
# #                 if j not in visited and calculate_iou(persons[current][:4], other[:4]) > iou_threshold:
# #                     visited.add(j)
# #                     stack.append(j)
# #                     cluster.append(other)

# #         clusters.append(cluster)
# #     return clusters

# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         print("Failed to grab frame")
# #         break

# #     # 객체 탐지
# #     results = model(frame)

# #     # 탐지된 사람 수 계산 및 위치 저장
# #     persons = [x for x in results.xyxy[0] if int(x[-1]) == 0 and x[4] >= confidence_threshold]  # class 0은 사람을 의미, 인식 정확도 0.3 이상
# #     num_persons = len(persons)

# #     # 탐지된 인원 수 화면에 출력
# #     cv2.putText(frame, f'Total Persons: {num_persons}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# #     # 사람들 간의 거리 계산 및 클러스터링
# #     clusters = find_clusters(persons, 0.5)  # IoU가 0.5 이상인 경우 클러스터로 간주

# #     for cluster in clusters:
# #         if len(cluster) >= alert_threshold:
# #             # 클러스터의 중심 좌표 계산
# #             centers = [get_center(person[:4]) for person in cluster]
# #             centers = np.array(centers)
# #             avg_center = np.mean(centers, axis=0)

# #             # 경고 영역 설정
# #             x_min = int(np.min(centers[:, 0]) - 20)
# #             y_min = int(np.min(centers[:, 1]) - 20)
# #             x_max = int(np.max(centers[:, 0]) + 20)
# #             y_max = int(np.max(centers[:, 1]) + 20)

# #             # 투명한 빨간 박스 그리기
# #             overlay = frame.copy()
# #             cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
# #             alpha = 0.4  # 투명도
# #             frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# #             # 경고 메시지 출력
# #             cv2.putText(frame, 'DANGER: Crowding Detected!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# #     # 결과 표시
# #     for *xyxy, conf, cls in persons:
# #         label = f'{model.names[int(cls)]} {conf:.2f}'
# #         cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
# #         cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# #     cv2.imshow('Frame', frame)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import torch
import cv2
import numpy as np
from flask import Flask, render_template, Response
import os

app = Flask(__name__)

# 모델 로드 (사전 훈련된 YOLOv5 모델 사용)
model_path = r'C:\Users\joyon\EyeSafer_AI\j-in\best.pt'  # 올바른 경로를 지정하세요
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# 비디오 경로 지정
video_path = r"C:\Users\joyon\EyeSafer_AI\testvideo\test_seoul.mp4"

# 경고 인원 수 설정
alert_threshold = 3  # 3명 이상일 경우 표시
confidence_threshold = 0.3  # 인식 정확도 0.3 이상인 경우만 사람으로 인식
distance_threshold = 50  # 가까운 거리의 기준 (픽셀 단위로 조정 가능)

def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 객체 탐지
        results = model(frame)

        # 탐지된 사람 수 계산 및 위치 저장
        persons = [x for x in results.xyxy[0] if int(x[-1]) == 0 and x[4] >= confidence_threshold]  # class 0은 사람을 의미, 인식 정확도 0.3 이상
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
                if distance < distance_threshold:  # 가까운 거리의 기준 (픽셀 단위로 조정 가능)
                    close_persons.append((box1, box2))

        # 일정 거리 이하로 모여 있는 사람 수가 alert_threshold 이상인 경우 경고 박스 표시
        if len(close_persons) >= alert_threshold:
            # 클러스터의 중심 좌표 계산
            centers = []
            for box1, box2 in close_persons:
                centers.append(((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2))
                centers.append(((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2))

            # 중심 좌표의 평균 계산
            centers = np.array(centers)
            avg_center = np.mean(centers, axis=0)

            # 경고 영역 설정
            x_min = int(np.min(centers[:, 0]) - 20)
            y_min = int(np.min(centers[:, 1]) - 20)
            x_max = int(np.max(centers[:, 0]) + 20)
            y_max = int(np.max(centers[:, 1]) + 20)

            # 투명한 빨간 박스 그리기
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
            alpha = 0.4  # 투명도
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # 경고 메시지 출력
            cv2.putText(frame, 'DANGER: Crowding Detected!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 결과 표시
        for *xyxy, conf, cls in persons:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
