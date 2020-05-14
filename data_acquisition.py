import serial
import matplotlib.pyplot as plt
import numpy as np
import glob
import csv

connected = False

#finds COM port that the Arduino is on (assumes only one Arduino is connected)
wmi = glob.glob("/dev/ttyACM*")
if len(wmi) == 1:
    comPort = wmi[0]
    print(f"{comPort} is Arduino")

ser = serial.Serial(comPort, 115200) #sets up serial connection (make sure baud rate is correct - matches Arduino)

while not connected:
    serin = ser.read()
    connected = True


plt.ion()                    #sets plot to animation mode

length = 1000                 #determines length of data taking session (in data points)
t = []
x_orien = [0]*length
y_orien = [0]*length
z_orien = [0]*length
x_acc = []              
y_acc = []
z_acc = []
x_gyr = []              
y_gyr = []
z_gyr = []
qw = []
qx = []
qy = []
qz = []

xline, = plt.plot(x_orien)         #sets up future lines to be modified
yline, = plt.plot(y_orien)
zline, = plt.plot(z_orien)
plt.ylim(-180,180)        #sets the y axis limits

for i in range(length):     #while you are taking data
    data = ser.readline()    #reads until it gets a carriage return. MAKE SURE THERE IS A CARRIAGE RETURN OR IT READS FOREVER
    sep = data.split()      #splits string into a list at the tabs
    try:
        if len(sep) == 14:
            #print sep
            t.append(float(sep[0]))
            x_acc.append(float(sep[1]))
            y_acc.append(float(sep[2]))
            z_acc.append(float(sep[3]))
            x_gyr.append(-float(sep[4]))
            y_gyr.append(-float(sep[5]))
            z_gyr.append(float(sep[6]))
            if -float(sep[7]) > -180:
                x_orien.append(-float(sep[7]))
            else:
                x_orien.append(-float(sep[7]) + 360)
            
            y_orien.append(float(sep[8]))
            z_orien.append(float(sep[9]))
            qw.append(float(sep[10]))
            qx.append(float(sep[11]))
            qy.append(float(sep[12]))
            qz.append(float(sep[13]))
            

        del x_orien[0]
        del y_orien[0]
        del z_orien[0]
    except :
        pass

    xline.set_xdata(np.arange(len(x_orien))) #sets xdata to new list length
    yline.set_xdata(np.arange(len(y_orien)))
    zline.set_xdata(np.arange(len(z_orien)))

    xline.set_ydata(x_orien)                 #sets ydata to new list
    yline.set_ydata(y_orien)
    zline.set_ydata(z_orien)

    xline.set_label("z")
    yline.set_label("y")
    zline.set_label("x")

    plt.legend()

    plt.pause(0.001)                   #in seconds
    plt.draw()                         #draws new plot


rows = list(zip(t, x_acc, y_acc, z_acc, x_gyr, y_gyr, z_gyr, x_orien, y_orien, z_orien, qw, qx, qy, qz))                  #combines lists together
row_arr = [list(a) for a in rows]               #creates array from list

with open("/home/huydung/devel/intern/Measurements/data.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(row_arr)

ser.close()     # closes serial connection (very important to do this! if you have an error partway through the code, 
                # type this into the cmd line to close the connection)