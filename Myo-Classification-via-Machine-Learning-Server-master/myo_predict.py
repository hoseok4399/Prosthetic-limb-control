from __future__ import print_function
import myo_train_cnn
from config import *
from SocketServer import SocketServer

from collections import deque
from threading import Lock, Thread
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
#np.random.seed(1)
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.models import load_model
from sklearn import preprocessing

import myo
import time
import sys
import psutil
import os
import serial

# This training set will contain 1000 samples of 8 sensor values
global training_set
global number_of_samples
global verification_set
global data_array
number_of_samples = 1000
verification_set = np.zeros((8, number_of_samples))
data_array = []


def find_one_hot(labels,classes):
    output = tf.one_hot(labels,classes,axis=0)
    sess = tf.Session()
    out = sess.run(output)
    sess.close
    return out

# This process checks if Myo Connect.exe is running
def check_if_process_running():
    try:
        for proc in psutil.process_iter():
            if proc.name()=='Myo Connect.exe':
                return True            
        return False
            
    except (psutil.NoSuchProcess,psutil.AccessDenied, psutil.ZombieProcess):
        print (PROCNAME, " not running")

# If the process Myo Connect.exe is not running then we restart that process
def restart_process():
    PROCNAME = "Myo Connect.exe"

    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == PROCNAME:
            proc.kill()
            # Wait a second
            time.sleep(1)

    while(check_if_process_running()==False):
        # 여기 수정
        path = MYO_PATH
        os.startfile(path)
        time.sleep(1)

    print("Process started")
    time.sleep(5) # 추가로 넣은 것, 연결 시간, 여기 수정
    return True



# This is Myo-python SDKs listener that listens to EMG signal
class Listener(myo.DeviceListener):
    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)

    def on_connected(self, event):
        print("Myo Connected")
        self.started = time.time()
        event.device.stream_emg(True)
        
    def get_emg_data(self):
        with self.lock:
            print("H")

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.emg))
            
            if len(list(self.emg_data_queue))>=number_of_samples:
                data_array.append(list(self.emg_data_queue))
                self.emg_data_queue.clear()
                return False

sock = SocketServer(HOST_PORT)

# This method is responsible for training EMG data
def Test():
    global verification_set
    global number_of_samples

    number_of_samples=1000 
    verification_set = np.zeros((8, number_of_samples))

    while(restart_process() != True):
        pass
    # Wait for 3 seconds until Myo Connect.exe starts
    time.sleep(3)
    
    # Initialize the SDK of Myo Armband
    # 여기 수정, myo64.dll이 있는 경로로
    myo.init(DLL_PATH)
    hub = myo.Hub()
    listener = Listener(number_of_samples)

    legend = ['Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6','Sensor 7','Sensor 8']

    name = input("Enter name of Subject that you used")

    
    #conc_array = np.loadtxt('C:/Users/gpdnj/Desktop/myo_nn1/' +
    #                        name+'_five_movements.txt', delimiter='\n')

    model = keras.models.load_model(
        MODEL_PATH + name+'_five_finger_model.h5')
    adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    averages = number_of_samples/50
    verification_averages = np.zeros((int(averages),8))

    while True:
        while True:
            try:
                print("Open your finger")
                hub = myo.Hub()        
                number_of_samples=200
                listener = Listener(number_of_samples)
                hub.run(listener.on_event,20000)

                # Here we send the received number of samples making them a list of 1000 rows 8 columns
                verification_set = np.array((data_array[0]))
                data_array.clear()

                verification_set = np.absolute(verification_set)

                div = 50
                # We add one because iterator below starts from 1
                batches = int(number_of_samples/div) + 1
                for i in range(1,batches):

                    verification_averages[i-1,:] = np.mean(verification_set[(i-1)*div:i*div,:],axis=0)

                verification_data = verification_averages
                verification_data = verification_data.reshape( verification_data.shape[0], verification_data.shape[1], 1)
                print("Verification matrix shape is " , verification_data.shape)
                
                predictions = model.predict(verification_data,batch_size=16)
                predicted_value = np.argmax(predictions[0])
                print(predictions[0])
                print(predicted_value)
                if predicted_value == 0:
                    print("Thumb open")
                elif predicted_value == 1:
                    print("Index finger open")
                elif predicted_value == 2:
                    print("Middle finger open")
                elif predicted_value == 3:
                    print("Ring finger open")
                elif predicted_value == 4:
                    print("Pinky finger open")
                elif predicted_value == 5:
                    print("Five fingers open")
                elif predicted_value == 6:
                    print("All fingers closed")
                #elif predicted_value == 7:
                #    print("Four fingers open")
                #elif predicted_value == 8:
                #    print("Five fingers open")
                #elif predicted_value == 9:
                #    print("All fingers closed")
                #elif predicted_value == 10:
                #    print("Grasp movement")
                #elif predicted_value == 11:
                #    print("Pick movement")
                else:
                    print("Undefined movement")

                sock.send(predicted_value)
                
                time.sleep(1.5)

            except:
                while(restart_process()!=True):
                    pass
                # Wait for 3 seconds until Myo Connect.exe starts
                time.sleep(3)
        

if __name__ == '__main__':
    Test()
