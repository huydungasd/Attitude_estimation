import numpy as np
import pandas as pd
import math
from time import sleep

class IMU:
        
    def __init__(self, file_path):

        self.data = pd.read_csv(file_path).to_numpy(dtype=float)
        
        print("[IMU] Initialised.")
    
    def get_t(self):
        return self.data[1:, 0]         
        
    # rad/s
    def get_gyro(self):
        return [self.data[1:, 7], self.data[1:, 8], self.data[1:, 9]]        

    def get_mag(self):
        return [self.data[1:, 4], self.data[1:, 5], self.data[1:, 6]]    
        
    # m/s^2
    def get_acc(self):
        return [self.data[1:, 1], self.data[1:, 2], self.data[1:, 3]]
    
    # rad
    def get_acc_angles(self):
        [ax, ay, az] = self.get_acc()
        phi = np.arctan2(ay, np.sqrt(ax ** 2.0 + az ** 2.0))
        theta = np.arctan2(-ax, np.sqrt(ay ** 2.0 + az ** 2.0))
        return [phi, theta]

def get_mag_yaw(mag_x, mag_y, mag_z, prev_angle):
    a = -mag_y * np.cos(prev_angle[0]) + mag_z * np.sin(prev_angle[0])
    b = mag_x * np.cos(prev_angle[1])
    c = mag_y * np.sin(prev_angle[1]) * np.sin(prev_angle[0])
    d = mag_z * np.sin(prev_angle[1]) * np.cos(prev_angle[0])

    return (np.arctan2(a,  (b + c + d)) + np.pi/2)