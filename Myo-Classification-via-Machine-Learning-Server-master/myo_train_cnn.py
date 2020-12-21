from __future__ import print_function
from config import *
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
from keras.applications import VGG16
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
global index_training_set, middle_training_set,thumb_training_set,verification_set
global data_array
number_of_samples = 2000
data_array=[]

Sensor1 = np.zeros((1,number_of_samples))
Sensor2 = np.zeros((1,number_of_samples))
Sensor3 = np.zeros((1,number_of_samples))
Sensor4 = np.zeros((1,number_of_samples))
Sensor5 = np.zeros((1,number_of_samples))
Sensor6 = np.zeros((1,number_of_samples))
Sensor7 = np.zeros((1,number_of_samples))
Sensor8 = np.zeros((1,number_of_samples))

index_open_training_set = np.zeros((8,number_of_samples))
middle_open_training_set = np.zeros((8,number_of_samples))
thumb_open_training_set = np.zeros((8,number_of_samples))
ring_open_training_set = np.zeros((8,number_of_samples))
pinky_open_training_set = np.zeros((8,number_of_samples))
# two_open_training_set = np.zeros((8,number_of_samples))
# three_open_training_set = np.zeros((8,number_of_samples))
# four_open_training_set = np.zeros((8,number_of_samples))
five_open_training_set = np.zeros((8,number_of_samples))
all_fingers_closed_training_set = np.zeros((8,number_of_samples))
# grasp_training_set = np.zeros((8,number_of_samples))
# pick_training_set = np.zeros((8,number_of_samples))

verification_set = np.zeros((8,number_of_samples))
training_set = np.zeros((8,number_of_samples))


thumb_open_label = 0
index_open_label = 1
middle_open_label = 2
ring_open_label = 3
pinky_open_label = 4
# two_open_label = 5
# three_open_label = 6
# four_open_label = 7
five_open_label = 5
all_fingers_closed_label = 6
# grasp_label = 10
# pick_label = 11

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



# This method is responsible for training EMG data
def Train(conc_array):
    global training_set
    global index_open_training_set, middle_open_training_set, thumb_open_training_set, ring_open_training_set, pinky_open_training_set, verification_set
    global two_open_training_set, three_open_training_set, four_open_training_set,five_open_training_set,all_fingers_closed_training_set,grasp_training_set,pick_training_set
    global number_of_samples

    verification_set = np.zeros((8,number_of_samples))
    print (number_of_samples)
    
    labels = []
        
    print(conc_array,conc_array.shape)

    # This division is to make the iterator for making labels run 30 times in inner loop and 10 times in outer loop running total 300 times for 10 finger movements
    samples = conc_array.shape[0]/7
    # Now we append all data in training label
    # We iterate to make 5 finger movement labels.
    for i in range(0,7):
        for j in range(0,int(samples)):
            labels.append(i)
    labels = np.asarray(labels)
    print(labels, len(labels),type(labels))
    print(conc_array.shape[0])
    permutation_function = np.random.permutation(conc_array.shape[0])

    total_samples = conc_array.shape[0]
    all_shuffled_data,all_shuffled_labels = np.zeros((total_samples,8)),np.zeros((total_samples,8))
        
    all_shuffled_data,all_shuffled_labels = conc_array[permutation_function],labels[permutation_function]
    print(all_shuffled_data.shape)
    print(all_shuffled_labels.shape)
    
    number_of_training_samples = np.int(np.floor(0.8*total_samples))        
    train_data = np.zeros((number_of_training_samples,8))
    train_labels = np.zeros((number_of_training_samples,8))
    print("TS ", number_of_training_samples, " S " , number_of_samples)
    number_of_validation_samples = np.int(total_samples-number_of_training_samples)
    train_data = all_shuffled_data[0:number_of_training_samples,:]
    train_labels = all_shuffled_labels[0:number_of_training_samples,]
    print("Length of train data is ", train_data.shape)
    validation_data = all_shuffled_data[number_of_training_samples:total_samples,:]
    validation_labels = all_shuffled_labels[number_of_training_samples:total_samples,]
    print("Length of validation data is ", validation_data.shape , " validation labels is " , validation_labels.shape)
    print(train_data,train_labels)  

    #number_of_training_samples
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    validation_data = validation_data.reshape(validation_data.shape[0],validation_data.shape[1], 1)
    print("check x: ", train_data.shape)
    print("check y: ", train_labels.shape)
    
    # 모델 수정하려면 여기 수정
    if os.path.isfile(MODEL_PATH + name+'_five_finger_model.h5'):
        model = keras.models.load_model(
        MODEL_PATH + name+'_five_finger_model.h5')
    else: 
        #model = VGG16(weights='imagenet',include_top=False, input_shape=(80, 8, 1)) 
        model = keras.models.Sequential()

        model.add(keras.layers.Conv1D(filters=32, kernel_size=2,activation = 'relu', input_shape=(8, 1)))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))

        model.add(keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Conv1D(filters=256, kernel_size=1, activation = 'relu'))
        model.add(keras.layers.Conv1D(filters=256, kernel_size=1, activation='relu'))
        #model.add(keras.layers.MaxPooling1D(pool_size=2))
        # 습관적으로 썼는데 dimension이 너무 작아서 풀링은 안하는 게 나을 듯

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(12,activation='softmax'))

    adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    #여기 수정, epochs = 300 => 600    
    history = model.fit(train_data, train_labels, epochs=600,
                        validation_data=(validation_data, validation_labels), batch_size=16)
    # 여기 수정
    model.save(MODEL_PATH + name+'_five_finger_model.h5')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # 여기서 KeyError가 난다면 accuracy => acc, val_accuracy => val_acc로 수정
    # tensorflow와 matplot의 버전 호환 문제임

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(MODEL_PATH+name+'_accuracy.png')
    plt.show()

    #=================================================================================================
    #=================================================================================================
    #=================================================================================================

def main():
    unrecognized_training_set = np.zeros((8,number_of_samples))
    index_open_training_set = np.zeros((8,number_of_samples))
    middle_open_training_set = np.zeros((8,number_of_samples))
    thumb_open_training_set = np.zeros((8,number_of_samples))
    ring_open_training_set = np.zeros((8,number_of_samples))
    pinky_open_training_set = np.zeros((8,number_of_samples))
    # two_open_training_set = np.zeros((8,number_of_samples))
    # three_open_training_set = np.zeros((8,number_of_samples))
    # four_open_training_set = np.zeros((8,number_of_samples))
    five_open_training_set = np.zeros((8,number_of_samples))
    all_fingers_closed_training_set = np.zeros((8,number_of_samples))
    # grasp_training_set = np.zeros((8,number_of_samples))
    # pick_training_set = np.zeros((8,number_of_samples))
    
    verification_set = np.zeros((8,number_of_samples))
    
    training_set = np.zeros((8,number_of_samples))
    # This function kills Myo Connect.exe and restarts it to make sure it is running
    # Because sometimes the application does not run even when Myo Connect process is running
    # So i think its a good idea to just kill if its not running and restart it

    while(restart_process()!=True):
        pass
    # Wait for 3 seconds until Myo Connect.exe starts
    time.sleep(3)
    
    # Initialize the SDK of Myo Armband
    # 여기 수정, myo64.dll이 있는 경로로
    myo.init(DLL_PATH)
    hub = myo.Hub()
    listener = Listener(number_of_samples)

    legend = ['Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6','Sensor 7','Sensor 8']

    ################## HERE WE GET TRAINING DATA FOR THUMB FINGER OPEN ########
    while True:
        try:
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            input("Open THUMB ")    
            hub.run(listener.on_event,20000)
            thumb_open_training_set = np.array((data_array[0]))
            print(thumb_open_training_set.shape)
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
           
    # Here we send the received number of samples making them a list of 1000 rows 8 columns just how we need to feed to tensorflow
    
    ################## HERE WE GET TRAINING DATA FOR INDEX FINGER OPEN ########
    while True:
        try:
            input("Open index finger")
            start_time = time.time()
            hub = myo.Hub()
            listener = Listener(number_of_samples)

            hub.run(listener.on_event,20000)
            # Here we send the received number of samples making them a list of 1000 rows 8 columns 
            index_open_training_set = np.array((data_array[0]))
            
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)

    ################## HERE WE GET TRAINING DATA FOR MIDDLE FINGER OPEN #################
    while True:
        try:
            input("Open MIDDLE finger")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            middle_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)

    # Here we send the received number of samples making them a list of 1000 rows 8 columns
        
    ################## HERE WE GET TRAINING DATA FOR RING FINGER OPEN ##########
    while True:
        try:
            input("Open Ring finger")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            ring_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)

    ################### HERE WE GET TRAINING DATA FOR PINKY FINGER OPEN ####################
    while True:
        try:
            input("Open Pinky finger")
            start_time = time.time()
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            pinky_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #  ################### HERE WE GET TRAINING DATA FOR TWO FINGER OPEN ####################
    # while True:
    #     try:

    #         input("Open Two fingers")
    #         hub = myo.Hub()
    #         listener = Listener(number_of_samples)
    #         hub.run(listener.on_event, 20000)
    #         two_open_training_set = np.array((data_array[0]))
    #         data_array.clear()
    #         break
    #     except:
    #         while(restart_process() != True):
    #             pass
    #         # Wait for 3 seconds until Myo Connect.exe starts
    #         time.sleep(3)

    # ################### HERE WE GET TRAINING DATA FOR THREE FINGER OPEN ####################
    # while True:
    #     try:
    #         input("Open Three fingers")
    #         hub = myo.Hub()
    #         listener = Listener(number_of_samples)
    #         hub.run(listener.on_event, 20000)
    #         three_open_training_set = np.array((data_array[0]))
    #         data_array.clear()
    #         break
    #     except:
    #         while(restart_process() != True):
    #             pass
    #         # Wait for 3 seconds until Myo Connect.exe starts
    #         time.sleep(3)

    # ################### HERE WE GET TRAINING DATA FOR THREE FINGER OPEN ####################
    # while True:
    #     try:
    #         input("Open Four fingers")
    #         hub = myo.Hub()
    #         listener = Listener(number_of_samples)
    #         hub.run(listener.on_event, 20000)
    #         four_open_training_set = np.array((data_array[0]))
    #         data_array.clear()
    #         break
    #     except:
    #         while(restart_process() != True):
    #             pass
    #         # Wait for 3 seconds until Myo Connect.exe starts
    #         time.sleep(3)

    ################### HERE WE GET TRAINING DATA FOR FIVE FINGER OPEN ####################
    while True:
        try:
            input("Open Five fingers")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event, 20000)
            five_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process() != True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)

    ################### HERE WE GET TRAINING DATA FOR ALL FINGERS CLOSED ####################
    while True:
        try:
            input("Make all fingers closed")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event, 20000)
            all_fingers_closed_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process() != True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)

    # ################### HERE WE GET TRAINING DATA FOR GRASP MOVEMENT ####################
    # while True:
    #     try:
    #         input("Make Grasp movement")
    #         hub = myo.Hub()
    #         listener = Listener(number_of_samples)
    #         hub.run(listener.on_event, 20000)
    #         grasp_training_set = np.array((data_array[0]))
    #         data_array.clear()
    #         break
    #     except:
    #         while(restart_process() != True):
    #             pass
    #         # Wait for 3 seconds until Myo Connect.exe starts
    #         time.sleep(3)

    # ################### HERE WE GET TRAINING DATA FOR PICK MOVEMENT ####################
    # while True:
    #     try:
    #         input("Make Pick movement")
    #         hub = myo.Hub()
    #         listener = Listener(number_of_samples)
    #         hub.run(listener.on_event, 20000)
    #         pick_training_set = np.array((data_array[0]))
    #         data_array.clear()
    #         break
    #     except:
    #         while(restart_process() != True):
    #             pass
    #         # Wait for 3 seconds until Myo Connect.exe starts
    #         time.sleep(3)

    # Absolute of finger open data
    thumb_open_training_set = np.absolute(thumb_open_training_set)
    index_open_training_set = np.absolute(index_open_training_set)
    middle_open_training_set = np.absolute(middle_open_training_set)
    ring_open_training_set = np.absolute(ring_open_training_set)
    pinky_open_training_set = np.absolute(pinky_open_training_set)
    # Absolute of finger close data
    #two_open_training_set = np.absolute(two_open_training_set)
    #three_open_training_set = np.absolute(three_open_training_set)
    #four_open_training_set = np.absolute(four_open_training_set)
    five_open_training_set = np.absolute(five_open_training_set)
    all_fingers_closed_training_set = np.absolute(
        all_fingers_closed_training_set)
    #grasp_training_set = np.absolute(grasp_training_set)
    #pick_training_set = np.absolute(pick_training_set)

    div = 50
    averages = int(number_of_samples/div)
    thumb_open_averages = np.zeros((int(averages), 8))
    index_open_averages = np.zeros((int(averages), 8))
    middle_open_averages = np.zeros((int(averages), 8))
    ring_open_averages = np.zeros((int(averages), 8))
    pinky_open_averages = np.zeros((int(averages), 8))
    #two_open_averages = np.zeros((int(averages), 8))
    #three_open_averages = np.zeros((int(averages), 8))
    #four_open_averages = np.zeros((int(averages), 8))
    five_open_averages = np.zeros((int(averages), 8))
    all_fingers_closed_averages = np.zeros((int(averages), 8))
    #grasp_averages = np.zeros((int(averages), 8))
    #pick_averages = np.zeros((int(averages), 8))

    # Here we are calculating the mean values of all finger open data set and storing them as n/50 samples because 50 batches of n samples is equal to n/50 averages
    for i in range(1, averages+1):
        thumb_open_averages[i-1, :] = np.mean(
            thumb_open_training_set[(i-1)*div:i*div, :], axis=0)
        index_open_averages[i-1, :] = np.mean(
            index_open_training_set[(i-1)*div:i*div, :], axis=0)
        middle_open_averages[i-1, :] = np.mean(
            middle_open_training_set[(i-1)*div:i*div, :], axis=0)
        ring_open_averages[i-1,
                           :] = np.mean(ring_open_training_set[(i-1)*div:i*div, :], axis=0)
        pinky_open_averages[i-1, :] = np.mean(
            pinky_open_training_set[(i-1)*div:i*div, :], axis=0)

        #two_open_averages[i-1,
        #                  :] = np.mean(two_open_training_set[(i-1)*div:i*div, :], axis=0)
        #three_open_averages[i-1, :] = np.mean(
        #    three_open_training_set[(i-1)*div:i*div, :], axis=0)
        #four_open_averages[i-1,
        #                   :] = np.mean(four_open_training_set[(i-1)*div:i*div, :], axis=0)
        five_open_averages[i-1,
                           :] = np.mean(five_open_training_set[(i-1)*div:i*div, :], axis=0)
        all_fingers_closed_averages[i-1, :] = np.mean(
            all_fingers_closed_training_set[(i-1)*div:i*div, :], axis=0)
        #grasp_averages[i-1,
        #               :] = np.mean(grasp_training_set[(i-1)*div:i*div, :], axis=0)
        #pick_averages[i-1,
        #              :] = np.mean(pick_training_set[(i-1)*div:i*div, :], axis=0)

     
    # Here we stack all the data row wise
    conc_array = np.concatenate([thumb_open_averages,index_open_averages,middle_open_averages,ring_open_averages,pinky_open_averages,five_open_averages,all_fingers_closed_averages],axis=0)
    print(conc_array.shape)
    # 여기 수정 
    np.savetxt(MODEL_PATH+name+'_five_movements.txt', conc_array, fmt='%i')
    # In this method the EMG data gets trained and verified
    Train(conc_array)

if __name__ == '__main__':
    name = input("Enter name of Subject : ")
    main()
