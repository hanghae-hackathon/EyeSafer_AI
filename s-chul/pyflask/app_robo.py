import cv2
import requests
from PIL import Image
import numpy as np
from scipy.spatial import distance
from flask import Flask, render_template, Response
import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from threading import Lock

app = Flask(__name__)

# 로보플로우 API 설정
api_url = "https://detect.roboflow.com/crowd-counting-dataset-w3o7w/2"
api_key = "api_key"

# 비디오 파일 설정
video_path = r"C:\Users\dltls\EyeSafer\testvideo\test6.mp4"
cap = cv2.VideoCapture(video_path)

# 빨간 박스 면적 저장
red_box_areas = []
lock = Lock()

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

# 빨간 박스(객체가 겹치는 부분) 정의
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

def generate_frames():
    frame_count = 0
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
            if frame_area > 0:
                danger_level = total_red_box_area / frame_area
                cv2.putText(frame, f"Danger Level: {danger_level:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if total_red_box_area / frame_area >= 0.1:
                cv2.putText(frame, "Warning: High crowd density!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Detected {num_people} people", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 빨간 박스 면적 저장
            with lock:
                red_box_areas.append((frame_count, total_red_box_area, danger_level))
            frame_count += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/graph')
def graph():
    with lock:
        frame_counts, red_areas, danger_levels = zip(*red_box_areas) if red_box_areas else ([], [], [])
    
    fig = Figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(frame_counts, red_areas, danger_levels, cmap='viridis')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Red Box Area')
    ax.set_zlabel('Danger Level')
    ax.set_title('Video Frame in 3D')

    output = io.BytesIO()
    fig.savefig(output, format='png')
    output.seek(0)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
