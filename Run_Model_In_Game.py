#This code enables GTA screengrabs to be recorded and keystrokes pressed while in game. 

import numpy as np
import os
from PIL import ImageGrab
import cv2
import time
from sys import stdout
from IPython.display import clear_output
import grabber
#from grabber import grabber
import threading
import matplotlib.pyplot as plt
from collections import Counter
from random import shuffle
import glob
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
import ctypes
#
from keras.optimizers import rmsprop_v2
from keras.models import Sequential
from keras.layers import TimeDistributed, LSTM, Flatten, Dense, InputLayer, MaxPooling2D, Dropout, Activation, Embedding, GRU, ConvLSTM2D
from keras.layers.convolutional import Convolution2D
from keras import optimizers
from keras.models import load_model
from keras import initializers
from pynput.keyboard import Key, Controller
import h5py
import log
from heapq import nlargest

selected_model = 'CNN+RNN' #SEE LINE ~330 fOR MODEL DIRECTORY PATH. 
from keras.backend import set_session
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#set_session(tf.compat.v1.Session(config=config))

# For controlling the game
from inputsHandler import select_key
from tkinter import *
import pyautogui as gui

def key_press(key): #Converting to hex format for keystrokes. 
    if key == 0:
        a = hex(ord('A'))
        #a = int(a,10)
        return 0x41
    if key == 1:
        a= hex(ord('D'))
        a = int(a,0)
        return a
    if key == 2:
        return hex(ord('W'))
    if key == 3:
        return hex(ord('S'))
    if key == 4:
        return hex(ord('AW'))
    if key == 5:
        return hex(ord('AS'))
    if key == 6:
        return hex(ord('DW'))
    if key == 7:
        return hex(ord('DS'))
    return 'none'


def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

show_current_control = True #It will show a windows with a message indicating if the car is currently be controlled by
                            #Network  or by a Human
    
show_whatAIsees = False #It will show the 5 images that the netowrk uses the predict the output 

enable_evasion = False #If the program detects that the car is not moving (for example because it is stuck facing a wall and
                        #the network is not able to return to the road) It will make the car move backwards for a second.
global grb
#grb = grabber(bbox=(1,26,1601,926))
def screen_record(method = 'ImameGrab'): #Take raw image of the 1600x900 game window and compress. 
    if method == 'ImameGrab':
        printscreen =  ImageGrab.grab(bbox=(1,26,1601,926))
        rawIMG = np.array(printscreen)
        generalIMG = cv2.resize(rawIMG, dsize=(240, 160), interpolation=cv2.INTER_CUBIC)
    
    elif method == 'grabber': #Alternate screengrab method, did not work as well as imagegrab. 
        global grb
        printscreen = None
        printscreen = grb.grab(printscreen)
        rawIMG = np.array(printscreen)
        generalIMG = cv2.resize(rawIMG, dsize=(240, 160), interpolation=cv2.INTER_CUBIC)
    
    return generalIMG   

global front_buffer
global back_buffer
front_buffer = np.zeros((240, 160), dtype=np.int8)
back_buffer = np.zeros((240, 160), dtype=np.int8)

global fps
fps = 0

def img_thread():
    global front_buffer
    global back_buffer
    global fps
    
    last_time = time.time()
    while True:
        front_buffer = screen_record()
        # Swap buffers
        front_buffer, back_buffer = back_buffer, front_buffer
        fps = int(1.0/(time.time()-last_time))
        last_time = time.time()
    return
    
def preprocess_image(image):
    proccessed_image = cv2.resize(image,(240,160))
    
    return proccessed_imagew

from game_control import PressKey, ReleaseKey
from getkeys import key_check

def keys_to_output(keys):
    '''
    '''
    output = [0,0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    if 'D' in keys:
        output[1] = 1
    if 'W' in keys:
        output[2] = 1
    if 'S' in keys:
        output[3] = 1
    
    return output

global seq
global num
num = 0
seq = []

global key_out
key_out = [0, 0, 0, 0]

def image_sequencer_thread():
    global back_buffer
    global seq
    global key_out
    global num
    
    # Frames per second capture rateSwSSwSSwS
    capturerate = 50.0
    while True:
        last_time = time.time()
        if len(seq) == 5:
            del seq[0]

        seq.append(preprocess_image(np.copy(back_buffer)))
        num = num + 1
        keys = key_check()
        key_out = keys_to_output(keys)
        waittime = (1.0/capturerate)-(time.time()-last_time)

def reshape_custom_X(data, verbose = 1): #Reshape data input
    reshaped = np.zeros((data.shape[0], 5, 160, 240, 3), dtype=np.float32)
    for i in range(0, data.shape[0]):
        for j in range(0, 5):
            if (verbose == 1):
                clear_output(wait=True)
                stdout.write('Reshaped image: ' + str(i))
                stdout.flush()
            reshaped[i][j] = data[i][j]/255.
            
    return reshaped


def get_num_batches(length, BATCH_SIZE):
    if (int(length/BATCH_SIZE)*BATCH_SIZE == length):
        return int(length/BATCH_SIZE)
    else:
        return int(length/BATCH_SIZE)+1

def get_start_end(iteration, BATCH_SIZE, max_length):
    start = iteration*BATCH_SIZE
    if (start > max_length):
        print("ERROR: Check iterations made! Must be wrong")
        return -1, -1
    end = (iteration+1)*BATCH_SIZE
    if (end > max_length):
        end = max_length
    return start, end

#The following classes assist with physically pressing the button in game. 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
  _fields_ = [("wVk", ctypes.c_ushort),
              ("wScan", ctypes.c_ushort),
              ("dwFlags", ctypes.c_ulong),
              ("time", ctypes.c_ulong),
              ("dwExtraInfor", PUL)]

class HardwareInput(ctypes.Structure):
  _fields_ = [("uMsg", ctypes.c_ulong),
              ("wParamL", ctypes.c_short),
              ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
  _fields_ = [("dx", ctypes.c_long),
              ("dy", ctypes.c_long),
              ("mouseData", ctypes.c_ulong),
              ("dwFlags", ctypes.c_ulong),
              ("time", ctypes.c_ulong),
              ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
  _fields_ = [("ki", KeyBdInput), 
              ("mi", MouseInput)]

class Input(ctypes.Structure):
  _fields_ = [("type", ctypes.c_ulong),
              ("ii", Input_I)]
def PressKey(hexKeyCode):
  extra = ctypes.c_ulong(0)
  ii_ = Input_I()
  ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
  x = Input( ctypes.c_ulong(1), ii_ )
  ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
  extra = ctypes.c_ulong(0)
  ii_ = Input_I()
  ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
  x = Input( ctypes.c_ulong(1), ii_ )
  ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def new_key_press(key):
    keyboard = Controller()
    if key == 0:
        #gui.keyDown('A')
        PressKey(0x11)
        PressKey(0x1E)
        time.sleep(.3)
        ReleaseKey(0x11)
        ReleaseKey(0x1E)
        #gui.keyUp('A')
        return('A')
    if key == 1:
        #gui.keyDown('D')
        PressKey(0x11)
        PressKey(0x20)
        time.sleep(.3)
        ReleaseKey(0x11)
        ReleaseKey(0x20)
        #gui.keyUp('D')
        return('D')
    if key == 2:
        #gui.keyDown('W')
        PressKey(0x11)
        time.sleep(.3)
        ReleaseKey(0x11)
        #gui.keyUp('W')
        return('W')
    if key == 3:
        #gui.keyDown('S')
        PressKey(0x11)
        time.sleep(.3)
        ReleaseKey(0x11)
        #gui.keyUp('S')
        return('S')
    if key == 4:
        #gui.keyDown('AW')
        PressKey(0x1E)
        PressKey(0x11)
        time.sleep(.3)
        ReleaseKey(0x1E)
        ReleaseKey(0x11)
        gui.keyUp('AW')
        return('AW')
    if key == 5:
        #gui.keyDown('A')
        #gui.keyDown('S')
        PressKey(0x1E)
        PressKey(0x1F)
        time.sleep(.3)
        ReleaseKey(0x1E)
        ReleaseKey(0x1F)
        #gui.keyUp('A')
        #gui.keyUp('S')
        return('AS')
    if key == 6:
        #gui.keyDown('D')
        #gui.keyDown('W')
        PressKey(0x20)
        PressKey(0x11)
        time.sleep(.3)
        ReleaseKey(0x20)
        ReleaseKey(0x11)
        #gui.keyUp('D')
        #gui.keyUp('W')
        return('DW')
    if key == 7:
        #gui.keyDown('D')
        #gui.keyDown('S')
        PressKey(0x20)
        PressKey(0x1F)
        time.sleep(.3)
        ReleaseKey(0x20)
        ReleaseKey(0x20)
        #gui.keyUp('D')
        #gui.keyUp('S')
        return('DS')
    return('NONE')


def run_IA():
    global fps
    global front_buffer
    global back_buffer
    global seq
    global key_out
    global num
    #Model being used (Name of model must be imput at top of code)
    model = load_model('C:\\Users\\austi\\Dropbox\\ISU Semester 8\\ME 592 ML CPS\\Final\\Final_Project_592\\Final_Project_592\\DATA5\\'+selected_model)
    
    training_data = []
    threads = list()
    th_img = threading.Thread(target=img_thread)
    th_seq = threading.Thread(target=image_sequencer_thread)
    threads.append(th_img)
    threads.append(th_seq)
    th_img.start()
    time.sleep(1)
    th_seq.start()
    time.sleep(1)

    last_num = 0
    last_time = time.time()
    
    if show_current_control:
        root = Tk()
        var = StringVar()
        var.set('AI Driving')
        l = Label(root, textvariable = var, fg='green', font=("Courier", 44))
        l.pack()

    if enable_evasion: #This was not implemented. Attempted to develop technique to reverse after hitting a wall. 
            score = mse(img_seq[0],img_seq[4])
            if score < 1000:
                if show_current_control:
                    var.set('Backing Up')
                    l.config(fg='blue')
                    root.update()
                select_key(4)
                time.sleep(1)
                if np.random.rand()>0.5:
                    select_key(6)
                else:
                    select_key(8)
                time.sleep(0.2)
                if show_current_control:
                    var.set('AI Driving')
                    l.config(fg='green')
                    root.update()
    
    while True:
        img_seq = seq.copy()
        while len(img_seq) != 5 or last_num==num:
            del img_seq
            img_seq = seq.copy()
        last_num = num
        array=[]
        array.append([img_seq[0],img_seq[1],img_seq[2],img_seq[3],img_seq[4]])
        #array.append([img_seq[0]])
        NNinput = np.array(array)
        #print(NNinput)
        x = reshape_custom_X(NNinput[:,0:5],0)
        p = model.predict(x)

        sorted = p.flatten()
        sorted.sort()

        best = sorted[-1]
        #best2 = sorted[-2]

        location_best = np.where(p == best)[1][0]
        if location_best == 3:
            secondbest = sorted[-2]
            location_best = np.where(p == secondbest)[1][0]
            
            
        #location_best2 = np.where(p == best2)[1][0]
        new_key_press(location_best)  #press the key of most likely output
        new_key_press(2)
        #new_key_press(location_best2)
        output = new_key_press(location_best)   #Store the key pressed for printing to console
        #output2 = new_key_press(location_best2)   #Store the key pressed for printing to console

        time_act = time.time()
        clear_output(wait=True)
        stdout.write('Recording at {} FPS \n'.format(fps))
        stdout.write('Keys pressed: ' + str(output) + '\n')
        stdout.write('Actions per second: ' + str(1/(time_act-last_time)) + '\n')
        last_time = time.time()

run_IA()