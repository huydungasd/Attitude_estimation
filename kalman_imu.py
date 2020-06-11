import os
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
from imu import *
from numpy import convolve
from time import sleep, time
from math import sin, cos, tan, pi



fname = "145.csv"
# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    """
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
    """
    R = Rotation.from_euler('xyz', theta).as_matrix()
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
Q1 = np.eye(4)*0.01
R1 = np.eye(2)

C2 = np.array([[1, 0]])
P2 = np.eye(2)
Q2 = np.eye(2)
R2 = np.eye(1)

thresshold = 0.1

state_estimate1 = np.array([[0], [0], [0], [0]])
state_estimate2 = np.array([[0], [0]])

phi_hat = np.zeros(imu.data.shape[0] - 1)
theta_hat = np.zeros(imu.data.shape[0] - 1)
gamma_hat = np.zeros(imu.data.shape[0] - 1)

# Orientation by iternal algorithm of IMU sensor
phi, theta, gamma = -imu.data[1:, 12], -imu.data[1:, 11], -imu.data[1:, 10]
for i in range(len(gamma)):
    if gamma[i] < -pi:
        gamma[i] += 2*pi
phi = movingaverage(phi, 10)
theta = movingaverage(theta, 10)
gamma = movingaverage(gamma, 10)

[ax, ay, az] = imu.get_acc()
[phi_acc, theta_acc] = imu.get_acc_angles()
phi_acc = movingaverage(phi_acc, window=10)
theta_acc = movingaverage(theta_acc, window=10)

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

# Calculate accelerometer offsets
N = 50
phi_offset = 0
theta_offset = 0
gamma_offset = 0
phi_true_offset = 0
theta_true_offset = 0
gamma_true_offset = 0
for i in range(N):
    phi_offset += phi_acc[i]
    theta_offset += theta_acc[i]
    gamma_offset += get_mag_yaw(mag_x[i], mag_y[i], mag_z[i], np.array([phi[i], theta[i]]))
    phi_true_offset += phi[i]
    theta_true_offset += theta[i]
    gamma_true_offset += gamma[i]
phi_offset = phi_offset / N - phi_true_offset / N
theta_offset = theta_offset / N - theta_true_offset / N
gamma_offset = gamma_offset / N - gamma_true_offset / N
print("Roll, Pitch calculated by acceleration offset: " + str(phi_offset) + "," + str(theta_offset))
print("Yaw calculated by magnetometer offset: " + str(gamma_offset))
sleep(1)

# Get accelerometer measurements and remove offsets
phi_acc -= phi_offset
theta_acc -= theta_offset

print("Running...")
t = imu.get_t()
t = t - t[0]
phi_hat[0], theta_hat[0], gamma_hat[0] = phi[0], theta[0], gamma[0]
phi_interg, theta_interg, gamma_interg = np.zeros(phi_hat.shape), np.zeros(phi_hat.shape), np.zeros(phi_hat.shape)
phi_interg[0], theta_interg[0], gamma_interg[0] = phi[0], theta[0], gamma[0]
list1, list2 = [], []

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
    R_01 = eulerAnglesToRotationMatrix(prev_angle_intergral)
    R_02 = R_01 @ R_12
    phi_interg[i+1], theta_interg[i+1], gamma_interg[i+1] = Rotation.from_matrix(R_02).as_euler('xyz')

    # Kalman filter
    A1 = np.array([ [1, -dt, 0, 0], \
                    [0, 1, 0, 0],   \
                    [0, 0, 1, -dt], \
                    [0, 0, 0, 1]])

    B1 = np.array([ [dt, 0],    \
                    [0, 0],     \
                    [0, dt],    \
                    [0, 0]])

    gyro_input = np.array([[phi_interg[i] - phi_interg[i-1]], [theta_interg[i] - theta_interg[i-1]]])/dt # Suppose that this is our measurement of angular velocity in inertial frame
    state_estimate1 = A1 @ state_estimate1 + B1 @ gyro_input
    

    measurement1 = np.array([[phi_acc[i]], [theta_acc[i]]])
    P1 = A1 @ P1 @ A1.T + Q1
    if 9.7 < np.linalg.norm(np.array([ax[i], ay[i], az[i]])) < 9.9 and np.linalg.norm(np.array([p[i], q[i], r[i]])) < thresshold:
        list1.append(i)
        y_tilde1 = measurement1 - C1 @ state_estimate1
        S1 = R1 + C1 @ P1 @ C1.T
        K1 = P1 @ C1.T @ np.linalg.inv(S1)
        state_estimate1 = state_estimate1 + K1 @ y_tilde1
        P1 = (np.eye(4) - K1 @ C1) @ P1 @ (np.eye(4) - K1 @ C1).T + K1 @ R1 @ K1.T
    else:
        list2.append(i)

    phi_hat[i+1] = state_estimate1[0]
    theta_hat[i+1] = state_estimate1[2]

    # Kalman filter
    A2 = np.array([[1, -dt], [0, 1]])
    B2 = np.array([[dt], [0]])

    gamma_input = np.array([[gamma_interg[i] - gamma_interg[i-1]]])/dt # Suppose that this is our measurement of angular velocity in inertial frame
    state_estimate2 = A2 @ state_estimate2 + B2 @ gamma_input

    measurement2 = get_mag_yaw(mag_x[i], mag_y[i], mag_z[i], prev_angle) - gamma_offset
    P2 = A2 @ P2 @ A2.T + Q2
    if 9.7 < np.linalg.norm(np.array([ax[i], ay[i], az[i]])) < 9.9 and np.linalg.norm(np.array([p[i], q[i], r[i]])) < thresshold:
        y_tilde2 = measurement2 - C2 @ state_estimate2
        S2 = R2 + C2 @ P2 @ C2.T
        K2 = P2 @ C2.T @ np.linalg.inv(S2)
        state_estimate2 = state_estimate2 + K2 @ y_tilde2
        P2 = (np.eye(2) - K2 @ C2) @ P2 @ (np.eye(2) - K2 @ C2).T + K2 @ R2 @ K2.T

    gamma_hat[i+1] = state_estimate2[0]

# Display results
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(t[list1], phi_hat[list1], '.b', label='$\hat{\phi}$ (Prediction)')
axs[0].plot(t[list2], phi_hat[list2], '.r', label='$\hat{\phi}$ (Prediction)')
axs[0].plot(t, phi, 'c', label='$\phi$ by internal algo of IMU sensor')
axs[0].plot(t, phi_interg, 'g', label='$\phi_{interg}$ (Intergration)')
axs[0].set_ylabel('Rad')
axs[0].legend()

axs[1].plot(t, theta_hat, 'b', label='$\hat{\\theta}$ (Prediction)')
axs[1].plot(t, theta, 'c', label='$\\theta$ by internal algo of IMU sensor')
axs[1].plot(t, theta_interg, 'g', label='$\\theta_{interg}$ (Intergration)')
axs[1].set_ylabel('Rad')
axs[1].legend()

axs[2].plot(t, gamma_hat, 'b', label='$\hat{\gamma}$ (Prediction)')
axs[2].plot(t, gamma, 'c', label='$\gamma$ by internal algo of IMU sensor')
axs[2].plot(t, gamma_interg, 'g', label='$\gamma_{interg}$ (Intergration)')
axs[2].set_ylabel('Rad')
axs[2].set_xlabel('Time')
axs[2].legend()
plt.show()