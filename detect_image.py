# Import necessary libraries.
from ultralytics import YOLO
import cv2
import os

# --- Step 1: Initialize the YOLO model ---
# We load the pre-trained 'yolov8n.pt' model.
# 'n' stands for nano, which is the smallest and fastest model.
# It's perfect for a project like this as it has a small download size and runs quickly.
# This model has been pre-trained on the COCO dataset, which includes the 'person' class.
model = YOLO('yolov8n.pt')

# --- Step 2: Define a folder and a list of images to process ---
# This example will process a single image. You can add more to the list.
# Ensure the image file (e.g., 'person.jpg') is in the same directory as this script.
# If you are using the Coco dataset images you downloaded, place them here.
image_path = 'person.jpg'  # Replace with the name of your image file

# Check if the image file exists.
if not os.path.exists(image_path):
    print(f"Error: The image file '{image_path}' was not found.")
    print("Please make sure you have an image file in the same folder and have updated the 'image_path' variable.")
else:
    # --- Step 3: Run the detection on the image ---
    # The 'conf' parameter sets the confidence threshold. A higher value means the model
    # must be more certain to detect an object. We'll use 0.5 (50%).
    # The 'verbose=False' flag prevents the model from printing a lot of output to the console.
    results = model(image_path, conf=0.5, verbose=False)

    # --- Step 4: Process and draw the results on the image ---
    # Load the image using OpenCV.
    image = cv2.imread(image_path)

    # Iterate through the detected results.
    for r in results:
        # Get the detected boxes.
        boxes = r.boxes
        
        # Iterate through each detected box.
        for box in boxes:
            # Check if the detected object is a person.
            # The 'cls' attribute holds the class ID. For the COCO dataset, 'person' is class 0.
            if int(box.cls[0]) == 0:
                # Get the coordinates of the bounding box.
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get the confidence score.
                confidence = box.conf[0].cpu().numpy()
                
                # Draw the bounding box.
                # cv2.rectangle(image, start_point, end_point, color, thickness)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                
                # Create a label with class name and confidence score.
                label = f'Person {confidence:.2f}'
                
                # Draw the label text.
                # cv2.putText(image, text, origin, font, fontScale, color, thickness)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Step 5: Display the image with detections ---
    # Display the image in a window named 'Human Detection'.
    cv2.imshow('Human Detection', image)

    # Wait for a key press to close the window.
    cv2.waitKey(0)

    # Close all OpenCV windows.
    cv2.destroyAllWindows()

# --- How to run the script: ---
# 1. Make sure you have a virtual environment set up and the libraries installed.
# 2. Place an image file (e.g., person.jpg) in the same directory.
# 3. Run the script from your terminal: 'python detect_image.py'