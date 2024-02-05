import cv2
import numpy as np
from KalmanFilter import KalmanFilter
from Detector import detect

# Initialize the Kalman Filter
kf = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)

# Create video capture object
cap = cv2.VideoCapture("/Users/amine/Downloads/video2.MOV")  # Change to video file path if not using webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object Detection
    centers = detect(frame)

    for center in centers:
        # Kalman Filter Predict
        kf.predict()

        # Update with detected centroid
        kf.update(center)

        # Visualize
        current_state = kf.get_current_state()
        cv2.circle(frame, (int(center[0]), int(center[1])), 2, (0, 255, 0), 2)  # Detected centroid
        cv2.rectangle(frame, (int(current_state[0]) - 15, int(current_state[1]) - 15),
                      (int(current_state[0]) + 15, int(current_state[1]) + 15), (255, 0, 0), 2)  # Predicted position
        cv2.rectangle(frame, (int(current_state[0]) - 10, int(current_state[1]) - 10),
                      (int(current_state[0]) + 10, int(current_state[1]) + 10), (0, 0, 255), 2)  # Updated (estimated) position

    # Show the frame
    cv2.imshow('Object Tracking', frame)

    # Break loop with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
