import cv2
from ultralytics import YOLO
# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")
# Open the video file
cap = cv2.VideoCapture(r"C:\Users\joyon\EyeSafer_AI\y-hyun\testvideo\test6.mp4")
assert cap.isOpened(), "Error reading video file"
# Get video properties: width, height, and frames per second (fps)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# Specify classes to count, for example: person (0)
classes_to_count = [0]  # Class IDs for person
# Threshold for the number of people in each area
people_threshold = 2
# List to store dynamic monitoring areas
monitor_areas = []
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
    # Initialize variables for monitoring area
    monitor_areas.clear()
    people_count = 0
    # Calculate monitoring areas dynamically based on detected people
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            if class_id in classes_to_count:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                monitor_areas.append((x1, y1, x2, y2))
                people_count += 1
                # Draw bounding box and label on the frame
                confidence = box.conf.item()
                label = f"{model.names[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Red rectangle
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Red text
    # Draw the monitoring areas if they exist
    for area in monitor_areas:
        cv2.rectangle(frame, (int(area[0]), int(area[1])),
                      (int(area[2]), int(area[3])), (0, 255, 0), 2)  # Green color
    # Check if people count exceeds threshold for each area
    for area in monitor_areas:
        people_in_area = sum(1 for area2 in monitor_areas if (area[0] <= area2[0] <= area[2] and area[1] <= area2[1] <= area[3]))
        if people_in_area >= people_threshold:
            cv2.rectangle(frame, (int(area[0]), int(area[1])),
                          (int(area[2]), int(area[3])), (0, 0, 255), -1)  # Red color
    # Display the total people count
    text = f"People count: {people_count}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Write the annotated frame to the output video
    video_writer.write(frame)
    # Show the frame (optional)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and writer objects
cap.release()
video_writer.release()
# Close all OpenCV windows
cv2.destroyAllWindows()