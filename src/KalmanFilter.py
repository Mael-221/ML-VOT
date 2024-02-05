import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])  
        self.x = np.array([[0], [0], [0], [0]])  

        # State transition matrix
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        # Control input matrix
        self.B = np.array([[(dt**2) / 2, 0], [0, (dt**2) / 2], [dt, 0], [0, dt]])
        
        # Observation matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # Process noise covariance
        self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0], [0, (dt**4)/4, 0, (dt**3)/2], [(dt**3)/2, 0, dt**2, 0], [0, (dt**3)/2, 0, dt**2]]) * std_acc**2
        
        # Measurement noise covariance
        self.R = np.array([[x_std_meas**2, 0], [0, y_std_meas**2]])
        
        # Initial estimation error covariance
        self.P = np.eye(4)

    def predict(self):
        # Predict state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        
        # Predict state covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Measurement update
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(4)
        self.P = (I - np.dot(K, self.H)) @ self.P

    def get_current_state(self):
        return self.x
