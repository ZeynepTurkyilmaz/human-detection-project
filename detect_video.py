# Import necessary libraries.
from ultralytics import YOLO
import cv2

# --- Step 1: Initialize the YOLO model ---
# We load the pre-trained 'yolov8n.pt' model, same as the image detection script.
# This model will be used for real-time video detection.
model = YOLO('yolov8n.pt')

# --- Step 2: Open the webcam feed ---
# cv2.VideoCapture(0) opens the default webcam (usually 0).
# If you have multiple cameras, you might need to change this number.
cap = cv2.VideoCapture(0)

# Check if the webcam was opened successfully.
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam successfully opened. Press 'q' to exit.")
    
    # --- Step 3: Loop through each frame of the video feed ---
    while True:
        # Read a frame from the webcam.
        ret, frame = cap.read()
        
        # If 'ret' is False, it means there was an issue reading the frame.
        if not ret:
            break
            
        # --- Step 4: Run the detection on the current frame ---
        # We perform detection on the live frame with the same settings as the image script.
        # This is the core of real-time object detection.
        results = model(frame, conf=0.5, verbose=False)
        
        # --- Step 5: Process and draw the results on the frame ---
        # We iterate through the detections for the current frame.
        for r in results:
            # Get the detected boxes.
            boxes = r.boxes
            
            # Iterate through each detected box.
            for box in boxes:
                # Check if the detected object is a person (class 0).
                if int(box.cls[0]) == 0:
                    # Get the bounding box coordinates.
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get the confidence score.
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Draw the bounding box.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                    
                    # Create the label text.
                    label = f'Person {confidence:.2f}'
                    
                    # Draw the label text.
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        # --- Step 6: Display the frame with detections ---
        # Display the frame in a window named 'Live Human Detection'.
        cv2.imshow('Live Human Detection', frame)
        
        # Check for the 'q' key press to exit the loop.
        # cv2.waitKey(1) waits for 1 millisecond.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Step 7: Release resources ---
    # Release the webcam capture object.
    cap.release()
    # Close all OpenCV windows.
    cv2.destroyAllWindows()

# --- How to run the script: ---
# 1. Make sure you have a webcam connected.
# 2. Run the script from your terminal: 'python detect_video.py'
# 3. The video feed will pop up. Press 'q' to close it.
