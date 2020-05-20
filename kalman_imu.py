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
C1 = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
P1 = np.eye(4)
Q1 = np.eye(4)*100
R1 = np.eye(2)*3

C2 = np.array([[1, 0]])
P2 = np.eye(2)
Q2 = np.eye(2)*50
R2 = np.eye(1)*0.1

state_estimate1 = np.array([[0], [0], [0], [0]])
state_estimate2 = np.array([[0], [0]])

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

[ax, ay, az] = imu.get_acc()
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

# Get gyro measurements and calculate Euler angle derivatives
[mag_x, mag_y, mag_z] = imu.get_mag()
mag_x = movingaverage(mag_x, 10)
mag_y = movingaverage(mag_y, 10)
mag_z = movingaverage(mag_z, 10)

print("Running...")
t = imu.get_t()
phi_hat[0], theta_hat[0], gamma_hat[0] = phi[0], theta[0], gamma[0]
phi_interg, theta_interg, gamma_interg = np.zeros(phi_hat.shape), np.zeros(phi_hat.shape), np.zeros(phi_hat.shape)
phi_interg[0], theta_interg[0], gamma_interg[0] = phi[0], theta[0], gamma[0]

for i in range(imu.data.shape[0] - 2):

    # Sampling time
    dt = t[i+1] - t[i]

    prev_angle = np.array([phi_hat[i], theta_hat[i], gamma_hat[i]])
    prev_angle_intergral = np.array([phi_interg[i], theta_interg[i], gamma_interg[i]])

    # Get measurement
    vel_rate = np.array([p[i], q[i], r[i]])
    angle_change = vel_rate * dt

    # Calculate the orientation in inertial frame at the time (t+1)
    R_12 = eulerAnglesToRotationMatrix(angle_change)
    alpha = np.array(prev_angle_intergral)
    R_01 = eulerAnglesToRotationMatrix(alpha)
    R_02 = R_01 @ R_12
    phi_interg[i+1] = math.atan2(R_02[2, 1], R_02[2, 2]) / math.pi * 180
    theta_interg[i+1] = math.atan2(-R_02[2, 0], np.sqrt(R_02[2, 1]**2 + R_02[2, 2]**2)) / math.pi * 180
    gamma_interg[i+1] = math.atan2(R_02[1, 0], R_02[0, 0]) / math.pi * 180

    # Kalman filter
    A1 = np.array([[1, -dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, -dt], [0, 0, 0, 1]])
    B1 = np.array([[dt, 0], [0, 0], [0, dt], [0, 0]])

    gyro_input = np.array([[phi_interg[i]], [theta_interg[i]]])/dt # Suppose that this is our measurement of angular velocity in inertial frame
    state_estimate1 = A1 @ state_estimate1 + B1 @ gyro_input
    

    measurement1 = np.array([[phi_acc[i]], [theta_acc[i]]])
    P1 = A1 @ P1 @ A1.T + Q1
    y_tilde1 = measurement1 - C1 @ state_estimate1
    S1 = R1 + C1 @ P1 @ C1.T
    K1 = P1 @ C1.T @ np.linalg.inv(S1)
    state_estimate1 = state_estimate1 + K1 @ y_tilde1
    P1 = (np.eye(4) - K1 @ C1) @ P1 @ (np.eye(4) - K1 @ C1).T + K1 @ R1 @ K1.T

    phi_hat[i+1] = state_estimate1[0]
    theta_hat[i+1] = state_estimate1[2]

    # Kalman filter
    A2 = np.array([[1, -dt], [0, 1]])
    B2 = np.array([[dt], [0]])

    gamma_input = np.array([[gamma_interg[i]]])/dt # Suppose that this is our measurement of angular velocity in inertial frame
    state_estimate2 = A2 @ state_estimate2 + B2 @ gamma_input

    a = -mag_y[i] * np.cos(prev_angle[0] * np.pi / 180) + mag_z[i] * np.sin(prev_angle[0] * np.pi / 180)
    b = mag_x[i] * np.cos(prev_angle[1] * np.pi / 180)
    c = mag_y[i] * np.sin(prev_angle[1] * np.pi / 180) * np.sin(prev_angle[0] * np.pi / 180)
    d = mag_z[i] * np.sin(prev_angle[1] * np.pi / 180) * np.cos(prev_angle[0] * np.pi / 180)

    measurement2 = (np.arctan2(a,  (b + c + d)) * 180 / np.pi + 90)
    P2 = A2 @ P2 @ A2.T + Q2
    y_tilde2 = measurement2 - C2 @ state_estimate2
    S2 = R2 + C2 @ P2 @ C2.T
    K2 = P2 @ C2.T @ np.linalg.inv(S2)
    state_estimate2 = state_estimate2 + K2 @ y_tilde2
    P2 = (np.eye(2) - K2 @ C2) @ P2 @ (np.eye(2) - K2 @ C2).T + K2 @ R2 @ K2.T

    gamma_hat[i+1] = state_estimate2[0]

# Display results
plt.figure()
plt.plot(t, phi_hat, label='phi_hat (Prediction)')
# plt.plot(t[:-1], phi_acc[:-1], label='phi_acc')
plt.plot(t, phi, label='phi by internal algo of IMU sensor')
plt.plot(t, phi_interg, label='phi_interg (Intergration)')
plt.legend()


plt.figure()
plt.plot(t, theta_hat, label='theta_hat (Prediction)')
# plt.plot(t[:-1], theta_acc[:-1], label='theta_acc')
plt.plot(t, theta, label='theta by internal algo of IMU sensor')
plt.plot(t, theta_interg, label='theta_interg (Intergration)')
plt.legend()


plt.figure()
plt.plot(t, gamma_hat, label='gamma_hat (Prediction)')
# plt.plot(t[:-1], gamma_acc[:-1], label='gamma_acc')
plt.plot(t, gamma, label='gamma by internal algo of IMU sensor')
plt.plot(t, gamma_interg, label='gamma_interg (Measurement)')
plt.legend()
plt.show()