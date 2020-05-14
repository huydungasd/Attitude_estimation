import os
import numpy as np
import math
import matplotlib.pyplot as plt

from imu import *
from numpy import convolve
from time import sleep, time
from math import sin, cos, tan, pi



fname = "data0.csv"
# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta_deg) :
    theta = np.array([math.radians(theta_deg[0]), math.radians(theta_deg[1]), math.radians(theta_deg[2])])
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = R_z @ R_y @ R_x
    return R

# Lissage des signaux
def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.zeros(values.shape)
    sma[:window-1] = values[:window-1]
    sma[window-1:] = np.convolve(values, weights, 'valid')
    return sma


dir_path = os.path.dirname(os.path.realpath(__file__))
imu = IMU(file_path= os.path.join(dir_path, fname))

# Initialise matrices and variables
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
P = np.eye(4)
Q = np.eye(4)*100
R = np.eye(2)

state_estimate = np.array([[0], [0], [0], [0]])

phi_hat = np.zeros(imu.data.shape[0] - 1)
theta_hat = np.zeros(imu.data.shape[0] - 1)
gamma_hat = np.zeros(imu.data.shape[0] - 1)

# Orientation by iternal algorithm of IMU sensor
phi, theta, gamma = -imu.data[1:, 12], -imu.data[1:, 11], -imu.data[1:, 10]
for i in range(len(gamma)):
    if gamma[i] < -180:
        gamma[i] += 360
phi = movingaverage(phi, 10)
theta = movingaverage(theta, 10)
gamma = movingaverage(gamma, 10)

[phi_acc, theta_acc] = imu.get_acc_angles()
phi_acc = movingaverage(phi_acc, window=10)
theta_acc = movingaverage(theta_acc, window=10)

# Calculate accelerometer offsets
N = 500
phi_offset = 0.0
theta_offset = 0.0
phi_true_offset = 0
theta_true_offset = 0
for i in range(N):
    phi_offset += phi_acc[i]
    theta_offset += theta_acc[i]
    phi_true_offset += phi[i]
    theta_true_offset += theta[i]
phi_offset = float(phi_offset) / float(N) - phi_true_offset / N
theta_offset = float(theta_offset) / float(N) - theta_true_offset / N
print("Accelerometer offsets: " + str(phi_offset) + "," + str(theta_offset))
sleep(2)

# Get accelerometer measurements and remove offsets
phi_acc -= phi_offset
theta_acc -= theta_offset

# Get gyro measurements and calculate Euler angle derivatives
[p, q, r] = imu.get_gyro()
p = movingaverage(p, 10)
q = movingaverage(q, 10)
r = movingaverage(r, 10)

print("Running...")
t = imu.get_t()
phi_hat[0], theta_hat[0], gamma_hat[0] = phi[0], theta[0], gamma[0]
phi_mes, theta_mes, gamma_mes = np.zeros(phi_hat.shape), np.zeros(phi_hat.shape), np.zeros(phi_hat.shape)
phi_mes[0], theta_mes[0], gamma_mes[0] = phi[0], theta[0], gamma[0]

for i in range(imu.data.shape[0] - 2):

    # Sampling time
    dt = t[i+1] - t[i]

    prev_angle = np.array([phi_mes[i], theta_mes[i], gamma_mes[i]])

    # Get measurement
    vel_rate = np.array([p[i], q[i], r[i]])
    angle_change = vel_rate * dt

    # Calculate the orientation in inertial frame at the time (t+1)
    R_12 = eulerAnglesToRotationMatrix(angle_change)
    alpha = np.array(prev_angle)
    R_01 = eulerAnglesToRotationMatrix(alpha)
    R_02 = R_01 @ R_12
    phi_mes[i+1] = math.atan2(R_02[2, 1], R_02[2, 2]) / math.pi * 180
    theta_mes[i+1] = math.atan2(-R_02[2, 0], np.sqrt(R_02[2, 1]**2 + R_02[2, 2]**2)) / math.pi * 180
    gamma_mes[i+1] = math.atan2(R_02[1, 0], R_02[0, 0]) / math.pi * 180

    # Kalman filter
    A = np.array([[1, -dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, -dt], [0, 0, 0, 1]])
    B = np.array([[dt, 0], [0, 0], [0, dt], [0, 0]])

    gyro_input = np.array([[phi_mes[i]], [theta_mes[i]]])/dt # Suppose that this is our measurement of angular velocity in inertial frame
    state_estimate = A @ state_estimate + B @ gyro_input
    P = A @ P @ A.T + Q

    measurement = np.array([[phi_acc[i]], [theta_acc[i]]])
    y_tilde = measurement - C @ state_estimate
    S = R + C @ P @ C.T
    K = P @ C.T @ np.linalg.inv(S)
    state_estimate = state_estimate + K @ y_tilde
    P = (np.eye(4) - K @ C) @ P

    phi_hat[i+1] = state_estimate[0]
    theta_hat[i+1] = state_estimate[2]
    gamma_hat[i+1] = gamma_mes[i+1]

# Display results
plt.figure()
plt.plot(t, phi_hat, label='phi_hat (Prediction)')
# plt.plot(t[:-1], phi_acc[:-1], label='phi_acc')
plt.plot(t, phi, label='phi by internal algo of IMU sensor')
plt.plot(t, phi_mes, label='phi_mes (Measurement)')
plt.legend()


plt.figure()
plt.plot(t, theta_hat, label='theta_hat (Prediction)')
# plt.plot(t[:-1], theta_acc[:-1], label='theta_acc')
plt.plot(t, theta, label='theta by internal algo of IMU sensor')
plt.plot(t, theta_mes, label='theta_mes (Measurement)')
plt.legend()


plt.figure()
plt.plot(t, gamma_hat, label='gamma_hat (Prediction)')
# plt.plot(t[:-1], gamma_acc[:-1], label='gamma_acc')
plt.plot(t, gamma, label='gamma by internal algo of IMU sensor')
plt.plot(t, gamma_mes, label='gamma_mes (Measurement)')
plt.legend()
plt.show()