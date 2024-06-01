import cv2
from ultralytics import YOLO
import numpy as np

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
cap = cv2.VideoCapture(r"C:\Users\joyon\EyeSafer_AI\y-hyun\testvideo\test6.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties: width, height, and frames per second (fps)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize the video writer to save the output video
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Process video frames in a loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    # Perform object detection on the current frame
    results = model(frame)

    # Initialize heatmap
    heatmap = np.zeros_like(frame[:, :, 0], dtype=np.float32)

    # Draw bounding box and generate heatmap
    for result in results:
        for obj in result:
            if obj["label"] == "person":
                # Get bounding box coordinates
                x1, y1, x2, y2 = obj["box"]
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Generate heatmap
                heatmap[int(y1):int(y2), int(x1):int(x2)] += 1

    # Normalize heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)

    # Overlay heatmap on frame
    result_frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    # Write the annotated frame to the output video
    video_writer.write(result_frame)

    # Show the frame (optional)
    cv2.imshow("Frame", result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
video_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
