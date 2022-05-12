import numpy as np
import os
import time
from sys import stdout
from IPython.display import clear_output
import matplotlib.pyplot as plt
from keras import optimizers

from keras.optimizers import rmsprop_v2
from keras.models import Sequential
from keras.layers import TimeDistributed, LSTM, Flatten, Dense, InputLayer, MaxPooling2D, Dropout, Activation, Embedding, GRU, ConvLSTM2D
from keras.layers.convolutional import Convolution2D
from keras import optimizers
from keras.models import load_model
from keras import initializers
from keras import layers
import h5py
import log
from heapq import nlargest


selected_model = 'CNN' #Selected model enables multiple models to be ran from same code. 
if selected_model == 'CNN+RNN': #CNN and RNN architecture
    model = Sequential()
    model.add(InputLayer(input_shape=(5, 80, 120, 3)))
    model.add(TimeDistributed(Convolution2D(32, (4,4), data_format='channels_last')))
    model.add(TimeDistributed(Activation('relu')))
    print(model.output_shape)
    model.add(TimeDistributed(Convolution2D(32, (4,4), data_format='channels_last')))
    model.add(TimeDistributed(Activation('relu')))
    print(model.output_shape)
    model.add(TimeDistributed(MaxPooling2D(pool_size=(5, 5), data_format='channels_last')))
    model.add(TimeDistributed(Dropout(0.25)))
    print(model.output_shape)
    model.add(TimeDistributed(Convolution2D(16, (3,3), data_format='channels_last')))
    model.add(TimeDistributed(Activation('relu')))
    print(model.output_shape)
    model.add(TimeDistributed(MaxPooling2D(pool_size=(5, 5), data_format='channels_last')))
    model.add(TimeDistributed(Dropout(0.25)))
    print(model.output_shape)
    model.add(TimeDistributed(Flatten()))
    print(model.output_shape)
    model.add(GRU(256, kernel_initializer=initializers.RandomNormal(stddev=0.001))) #128
    #model.add(Dropout(0.25))
    print(model.output_shape)

    model.add(Dense(100))
    print(model.output_shape)

    model.add(Dense(80))
    print(model.output_shape)

    model.add(Dense(40))
    print(model.output_shape)

    model.add(Dense(9, activation='sigmoid'))
    print(model.output_shape)

    #opt = keras.optimizers.RMSprop(learning_rate=0.001)
    #opt = optimizers.rmsprop(lr=0.001)
    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy']) 


if selected_model == 'CNN': #Basic CNN architecture. 
    model = Sequential()
    model.add(InputLayer(input_shape=(5, 80, 120, 3))) #Establish input data shape.
    model.add(TimeDistributed(layers.Conv2D(32, (4, 4), activation='relu',data_format='channels_last'))) 
    #TimeDistributed enables the batches of 5 images for each keystroke to be used in model. 
    model.add(TimeDistributed(layers.MaxPooling2D(pool_size=(5, 5), data_format='channels_last')))
    model.add(TimeDistributed(layers.Conv2D(32, (4, 4), activation='relu', data_format='channels_last')))
    model.add(TimeDistributed(layers.MaxPooling2D(pool_size=(5, 5), data_format='channels_last')))
    model.add(TimeDistributed(layers.Conv2D(64, (2, 2), activation='relu', data_format='channels_last')))
    model.add(TimeDistributed(layers.Flatten()))
    model.add(layers.Flatten())
    model.add(layers.Dense(9, activation='relu'))
    #model.add(TimeDistributed(layers.Dense(9)))
    #opt = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer='adam',
                  #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  loss='BinaryCrossentropy',
                  metrics=['accuracy'])


def reshape_custom_X(data, verbose = 1): #reshapes the 255 based color data for analysis. 
#MUST CHANGE OF ARRAY TO SIZE OF INPUT IMAGE (Currently 80x120.)
    reshaped = np.zeros((data.shape[0], 5, 80, 120, 3), dtype=np.float32)
    for i in range(0, data.shape[0]):
        for j in range(0, 5):
            if (verbose == 1):
                clear_output(wait=True)
                stdout.write('Reshaped image: ' + str(i))
                stdout.flush()
            reshaped[i][j] = data[i][j]/255.
            
    return reshaped

def reshape_custom_y(data): #Reshapes input keystrokes from [A,D,W,S] input containing a combination of...
#inputs to a zero array of every possible combination with actual keystroke assigned a '1'. 
    reshaped = np.zeros((data.shape[0], 9), dtype=np.float32)
    for i in range(0, data.shape[0]):
            if np.array_equal(data[i][0] , [0,0,0,0]):
                reshaped[i][0] = 1.
            elif np.array_equal(data[i][0] , [1,0,0,0]):
                reshaped[i][1] = 1.
            elif np.array_equal(data[i][0] , [0,1,0,0]):
                reshaped[i][2] = 1.
            elif np.array_equal(data[i][0] , [0,0,1,0]):
                reshaped[i][3] = 1.
            elif np.array_equal(data[i][0] , [0,0,0,1]):
                reshaped[i][4] = 1.
            elif np.array_equal(data[i][0] , [1,0,1,0]):
                reshaped[i][5] = 1.
            elif np.array_equal(data[i][0] , [1,0,0,1]):
                reshaped[i][6] = 1.
            elif np.array_equal(data[i][0] , [0,1,1,0]):
                reshaped[i][7] = 1.
            elif np.array_equal(data[i][0] , [0,1,0,1]):
                reshaped[i][8] = 1.
    return reshaped

def get_num_batches(length, BATCH_SIZE): #Determine number of batches to run per epoch. 
    if (int(length/BATCH_SIZE)*BATCH_SIZE == length):
        return int(length/BATCH_SIZE)
    else:
        return int(length/BATCH_SIZE)+1

def get_start_end(iteration, BATCH_SIZE, max_length): #Determine bounds of batches. 
    start = iteration*BATCH_SIZE
    if (start > max_length):
        print("ERROR: Check iterations made! Must be wrong")
        return -1, -1
    end = (iteration+1)*BATCH_SIZE 
    if (end > max_length):
        end = max_length
    return start, end


log.openlog()
# Define the sizes and epoches
BATCH_SIZE = 10 #DEFINE TRAINING BATCH SIZE HERE
TEST_BATCH_SIZE = 10 #DEFINE TEST BATCH SIZE HERE
n_epochs = 2 #SELECT NUMBER OF EPOCHS HERE

# This path will be the path used to load the dataset files!
#Must direct to the .npz format file. 
files = np.load('C:\\Users\\austi\\Dropbox\\ISU Semester 8\\ME 592 ML CPS\\Final\\Final_Project_592\\Final_Project_592\\DATA4\\BriosoSmallImg500.npz', allow_pickle=True)

actual_file = 0
acc_for_files = []
for fil in files: #Runs loop for each training instance of 5 images and a keyboard input. 
    actual_file = actual_file + 1
    log.output("\n Loading input: "+fil)
    with files as data:
         training_data = data['arr_0']#Define training data from files. 

    #print("\n Balancing data...")
    #number = number_instances_per_class(training_data)
    #training_data = np.array(balance_data(number))

    log.output("\n Reshaping data...")
    np.random.shuffle(training_data) #Shuffle data to remove temporal element between each batch of data. 
    train = training_data[0:int(len(training_data)*0.90)] #Training data is 90% of total data. 
    test = training_data[int(len(training_data)*0.90):len(training_data)] # Testing is 10% of data. 
    del training_data
    #Split data into the images and the keyboard input. 
    X_train = reshape_custom_X(train[:, 0:5])
    y_train = reshape_custom_y(train[:, 5:6])
    X_test = reshape_custom_X(test[:, 0:5])
    y_test= reshape_custom_y(test[:, 5:6])
    del train, test
    log.output("\n Training...")
    data_length = len(X_train)
    n_batch = get_num_batches(data_length, BATCH_SIZE)
    for epoch in range(n_epochs): #Loop training for each epoch. 
        for iteration in range(n_batch): 
            start, end = get_start_end(iteration, BATCH_SIZE, data_length) 
            model.fit(x=X_train[start:end], y=y_train[start:end], epochs=1, verbose=0)
            history = model.fit(x=X_train[start:end], y=y_train[start:end], epochs=1, verbose=0)
            clear_output(wait=True) 
            log.output('\n => File : ' + str(actual_file) + ' of ' + str(len(files)))
            log.output('\n ==> EPOCH : ' + str(epoch+1) + ' of ' + str(n_epochs))
            log.output('\n ===> Iteration: ' + str(iteration+1) + ' of ' + str(n_batch))
            #score = model.evaluate(X_train[start:end], y_train[start:end], verbose=0)
            #log.output("\n Train batch accuracy: %.2f%%" % (score[1]*100))
            stdout.flush()
        prec = np.zeros((9))
        for i in range(len(y_test)): #Displays model performance for each epoch (which keystrokes were selected for each test image)
            p = model.predict(X_test[i:i+1]).astype(int)
            prec[p] = prec[p] + 1
        log.output("\n ==> Predictions:"+str(prec))
        stdout.flush()   
    total_acc = 0.0
    num = 0
    data_length = len(X_test)
    n_batch = get_num_batches(data_length, TEST_BATCH_SIZE)
    for iteration in range(n_batch):
        start, end = get_start_end(iteration, TEST_BATCH_SIZE, data_length)
        score = model.evaluate(X_test[start:end], y_test[start:end], verbose=0)
        log.output("\n => Batch %s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        total_acc = total_acc + score[1]
        num = num + 1
    total_acc = total_acc/float(num)
    log.output("\n ==> Total acc for file %s: %.2f%%" % (fil, total_acc*100))
    acc_for_files.append(total_acc*100)
    

    prec = np.zeros((9))
    for i in range(len(y_test)):
        p = model.predict(X_test[i:i+1]).astype(int)
        prec[p] = prec[p] + 1
    log.output("\n ==> Predictions:"+str(prec))
    
 

for acc in range(len(acc_for_files)):
    log.output("\n ==> Total acc after file %d: %.2f%%" % (acc, acc_for_files[acc]))

log.closelog()

model.save('C:\\Users\\austi\\Dropbox\\ISU Semester 8\\ME 592 ML CPS\\Final\\Final_Project_592\\Final_Project_592\\DATA4\\'+selected_model)