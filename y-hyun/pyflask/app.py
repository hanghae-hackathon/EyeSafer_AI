import cv2
import requests
from PIL import Image
import numpy as np
from scipy.spatial import distance
from flask import Flask, render_template, Response
import io
import matplotlib.pyplot as plt
app = Flask(__name__)

# 로보플로우 API 설정
api_url = "https://detect.roboflow.com/crowd-counting-dataset-w3o7w/2"
api_key = "w3ZNODb5rmLqLjKh9MVm"

# 비디오 파일 설정
# origin_cap = cv2.VideoCapture(r"C:\Users\joyon\EyeSafer_AI\y-hyun\testvideo\test6.mp4") 
origin_cap = cv2.VideoCapture(0) 
# detection_cap = cv2.VideoCapture(r"C:\Users\joyon\EyeSafer_AI\y-hyun\testvideo\test6.mp4")
detection_cap = origin_cap

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_crowd')
def video_crowd():
    return Response(generate_frames(detection_cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_origin')
def video_origin():
    return Response(generate_origin_frames(origin_cap), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
