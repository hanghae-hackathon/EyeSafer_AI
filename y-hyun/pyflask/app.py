import cv2
import requests
from PIL import Image
import torch
import numpy as np
from scipy.spatial import distance
from flask import Flask, render_template, Response
import io
import matplotlib.pyplot as plt
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
app = Flask(__name__)

# 로보플로우 API 설정
api_url = "https://detect.roboflow.com/crowd-counting-dataset-w3o7w/2"
api_key = "w3ZNODb5rmLqLjKh9MVm"

# 비디오 파일 설정
origin_cap = cv2.VideoCapture(r"C:\Users\dltls\EyeSafer\testvideo\test.mp4") 
# origin_cap = cv2.VideoCapture(0) 
detection_cap = cv2.VideoCapture(r"C:\Users\dltls\EyeSafer\testvideo\test.mp4")
# detection_cap = origin_cap
model3d_cap = cv2.VideoCapture(r"C:\Users\dltls\EyeSafer\testvideo\test.mp4")
jin_cap = cv2.VideoCapture(r"C:\Users\dltls\EyeSafer\testvideo\test.mp4")

def infer_frame(frame, confidence=0.20):
    # OpenCV 이미지를 PIL 이미지로 변환
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 이미지를 바이트 배열로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # API 호출 URL 구성
    infer_url = f"{api_url}?api_key={api_key}&confidence={confidence}"
    
    # 이미지 바이트 배열을 사용해 API 호출
    response = requests.post(infer_url, files={"file": img_byte_arr})
    
    return response.json()

def get_center(bbox):
    x_center, y_center, bbox_width, bbox_height = bbox
    return x_center, y_center

def draw_red_box(frame, points):
    x_coords = [pt[0] for pt in points]
    y_coords = [pt[1] for pt in points]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return (x_max - x_min) * (y_max - y_min)

def generate_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = infer_frame(frame, confidence=0.3)
        if 'predictions' in result:
            predictions = result['predictions']
            centers = []
            for prediction in predictions:
                x = int(prediction['x'])
                y = int(prediction['y'])
                width = int(prediction['width'])
                height = int(prediction['height'])
                x_min = x - width // 2
                y_min = y - height // 2
                x_max = x + width // 2
                y_max = y + height // 2
                centers.append((x, y))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            num_people = len(predictions)
            frame_height, frame_width, _ = frame.shape
            frame_area = frame_width * frame_height
            total_red_box_area = 0
            if len(centers) >= 3:
                dist_matrix = distance.cdist(centers, centers, 'euclidean')
                for i in range(len(dist_matrix)):
                    close_points = [centers[j] for j in range(len(dist_matrix)) if dist_matrix[i][j] < 50 and i != j]
                    if len(close_points) >= 2:
                        close_points.append(centers[i])
                        red_box_area = draw_red_box(frame, close_points)
                        total_red_box_area += red_box_area
            if total_red_box_area / frame_area >= 0.1:
                cv2.putText(frame, "Warning: High crowd density!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Detected {num_people} people", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_origin_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_3d_frames(cap):
        # 3D 그래프 생성
    fig = plt.figure() 
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
        output = io.BytesIO()
        fig.savefig(output, format='png')
        output.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + output.getvalue() + b'\r\n')
    # 비디오 캡처 객체 해제
    cap.release()

    # 창 닫기
    plt.close()
# 모델 로드 (사전 훈련된 YOLOv5 모델 사용)
model_path = r'C:\Users\dltls\EyeSafer\j-in\best.pt'  # 올바른 경로를 지정하세요
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)

# 경고 인원 수 설정
alert_threshold = 3  # 3명 이상일 경우 표시
confidence_threshold = 0.3  # 인식 정확도 0.3 이상인 경우만 사람으로 인식
distance_threshold = 50  # 가까운 거리의 기준 (픽셀 단위로 조정 가능)

def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

def generate_jin_frames(cap):
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

@app.route('/video_crowd')
def video_crowd():
    return Response(generate_frames(detection_cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_origin')
def video_origin():
    return Response(generate_origin_frames(origin_cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_3d')
def video_3d():
    return Response(generate_3d_frames(model3d_cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_jin')
def video_jin():
    return Response(generate_jin_frames(jin_cap), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
