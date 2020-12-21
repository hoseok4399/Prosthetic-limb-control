
from __future__ import print_function
import myo_lstm_predict
from config import *

from collections import deque
from threading import Lock, Thread

import myo
import time
import sys
import psutil
import os
import serial

# This method is responsible for training EMG data
def Ble():

     #### Here i send the predicted value to Arduino via Bluetooth so that it can open appropriate fingers ####

        # While 1 is used because sometimes bluetooth port throws exception in opening the COM Port
        # So i keep trying until the data is sent and confirmation received.
         while(1):
             try:
                 # Bluetooth at COM6
                 serialPort = serial.Serial(port=BLE_PORT,baudrate=BLE_BAUDRATE,bytesize=8,timeout=2,stopbits=serial.STOPBITS_ONE)                
                 value_to_bluetooth = str(predicted_value).encode()
                 serialPort.write(value_to_bluetooth)
                 time.sleep(1)
                 if serialPort.in_waiting>0:
                     serialString = serialPort.readline()
                     print(serialString)
                     # If we receive what we sent from Arduino bluetooth then all OK else bad value
                     if serialString == value_to_bluetooth:
                         print("Received")
                     else:
                         print("Bad value")
                 serialPort.close()
                 break
             except serial.SerialException as e:
                 #There is no new data from serial port
                 print (str(e))
             except TypeError as e:
                 print (str(e))
                 ser.port.close()
           

if __name__ == '__main__':
    Ble()
