import time
import torch
import numpy as np
import cv2
from plyer import notification

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Set confidence threshold
confidence_threshold = 0.5

# Initialize video capture (0 for default camera)
cap = cv2.VideoCapture(0)

# Function to send a break reminder
def send_break_reminder():
    """Function to send a reminder to take a break."""
    notification.notify(
        title="Break Reminder",
        message="It's time to take a short break! Stretch and relax.",
        timeout=10
    )

# Function to send a posture warning
def send_posture_warning():
    """Function to send a posture warning."""
    notification.notify(
        title="Posture Warning",
        message="Please straighten your back!",
        timeout=10
    )

# Track time for break reminder
last_break_time = time.time()
break_interval = 1800  # 30 minutes for a break reminder

while cap.isOpened():
    start = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)

    # Run the YOLO model on the frame
    results = model(frame)

    # Filter results by confidence threshold
    labels = results.names  # Get label names for detected objects
    boxes = results.xyxy[0].cpu().numpy()  # Get bounding boxes (xyxy format)
    confidences = results.pandas().xywh[0]['confidence'].values  # Get confidence scores

    # Draw bounding boxes and confidence on the frame
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls = box  # Extract bounding box coordinates, confidence, and class
        
        # Only process boxes with confidence above the threshold
        if conf >= confidence_threshold and int(cls) == 0:  # Only detect people (class 0 in COCO dataset)
            label = labels[int(cls)]  # Get the class label
            confidence = f'{conf:.2f}'  # Format confidence score

            # Draw the rectangle and label with confidence
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Draw bounding box
            cv2.putText(frame, f'{label} {confidence}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Check posture based on bounding box (simple logic)
            box_height = y2 - y1  # Height of the detected bounding box
            box_width = x2 - x1  # Width of the detected bounding box

            # Check if the bounding box's aspect ratio indicates proper sitting posture
            aspect_ratio = box_height / box_width

            if aspect_ratio < 1.5:  # This indicates the person is upright (normal sitting)
                print("Proper sitting posture detected!")
            else:  # If aspect ratio is too large, this may indicate slouching
                print("Slouching detected!")
                send_posture_warning()

    # Show the frame with detections
    cv2.imshow('Screen', frame)

    # Check if it's time to remind the user to take a break
    if time.time() - last_break_time > break_interval:
        send_break_reminder()
        last_break_time = time.time()

    # Exit if 'x' is pressed or if the window is closed
    if cv2.waitKey(10) & 0xFF == ord('x'):
        break
    if cv2.getWindowProperty("Screen", cv2.WND_PROP_VISIBLE) < 1:
        break

    # Calculate and display FPS
    end = time.time()
    fps = 1 / (end - start)
    print(f"FPS: {fps:.2f}")

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
