import cv2
import numpy as np
import os
import time

# Load the reference face images
reference_faces_dir = 'JupyterNB/roomates'
reference_faces = []

for filename in os.listdir(reference_faces_dir):
    img_path = os.path.join(reference_faces_dir, filename)
    reference_face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    reference_faces.append(reference_face)
    print(img_path)

# Initialize the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera, change as needed

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Initialize a counter for unregistered faces
    unregistered_face_count = 0

    for (x, y, w, h) in faces:
        # Crop the detected face region
        detected_face = gray[y:y + h, x:x + w]

        # Initialize a flag to check if the face is registered
        registered_face_detected = False

        # Iterate through reference faces and compare
        for reference_face in reference_faces:
            # Resize the reference face to match the detected face size
            reference_face = cv2.resize(reference_face, (w, h))

            # Compare the two faces using a simple difference threshold
            difference = cv2.absdiff(detected_face, reference_face)
            mean_difference = np.mean(difference)

            if mean_difference < 100:  # Adjust the threshold as needed
                registered_face_detected = True
                break

        # If no registered face is detected, increment the unregistered face count and capture the time
        if not registered_face_detected:
            unregistered_face_count += 1
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')

            # Display the alert message with a smaller font size
            alert_message = f'Alert: Unregistered Face Detected at {current_time}'
            cv2.putText(frame, alert_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()