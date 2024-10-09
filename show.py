import cv2
import supervision as sv
from ultralytics import YOLOv10
import os
import time

model_path = f'{os.getcwd()}/yolov10/yolov10x.pt'
model = YOLOv10(model_path)

video_path = f'{os.getcwd()}/videos/VID20240912150311.mp4'

# Load the video
cap = cv2.VideoCapture(video_path)

# Define colors for different types of objects
colors = {
    'car': (0, 255, 0),  # Green
    'bus': (255, 0, 0),  # Blue
    'truck': (0, 0, 255),  # Red
    'motorcycle': (255, 255, 0),  # Cyan
    'vehicle': (0, 255, 255),  # Yellow
    'ambulance': (255, 0, 255),  # Magenta
    'fire truck': (128, 0, 128),  # Purple
    'police car': (0, 128, 128),  # Teal
    'fire brigade': (128, 128, 0),  # Olive
    'person': (0, 128, 0)  # Dark Green
}

scren_shot_path = f'{os.getcwd()}/screenshots'
if not os.path.exists(scren_shot_path):
    os.makedirs(scren_shot_path)

screenshot_time = time.time()
# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the model on the frame and extract results
    results = model.predict(source=frame, conf=0.1, iou=0.45)
    total_vehicle_count = 0
    total_person_count = 0
    emergency_vehicle_count = 0

    # Process the detection results
    detected_boxes = results[0].boxes
    if detected_boxes:
        for box in detected_boxes:
            class_id = int(box.cls[0])  # Get the class ID of the detected object
            class_name = results[0].names[class_id]  # Get the class name using the ID

            print(f"Class: {class_name}")
            if class_name in ['car', 'bus', 'truck', 'motorcycle', 'vehicle']:
                total_vehicle_count += 1
            elif class_name in ['ambulance', 'fire truck', 'police car', 'fire brigade']:
                emergency_vehicle_count += 1
            elif class_name in ['person', 'human']:
                total_person_count += 1
            else:
                print(f"Class {class_name} not recognized")

    print(f"Total Vehicle Count: {total_vehicle_count}")
    print(f"Total Person Count: {total_person_count}")
    print(f"Emergency Vehicle Count: {emergency_vehicle_count}")

    # Draw the bounding boxes on the frame
    if detected_boxes:
        for box in detected_boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract the bounding box coordinates
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            color = colors.get(class_name, (255, 255, 255))  # Default to white if class not found
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    # Draw the count on the frame with different colors
    cv2.putText(frame, f"Total Vehicle Count: {total_vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Person Count: {total_person_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Emergency Vehicle Count: {emergency_vehicle_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize the frame
    frame = cv2.resize(frame, (1600, 900))

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Take a screenshot every 5 seconds
    if time.time() - screenshot_time > 5:
        screenshot_time = time.time()
        screenshot_path = f'{scren_shot_path}/screenshot_{time.time()}.png'
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved at {screenshot_path}")



# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()