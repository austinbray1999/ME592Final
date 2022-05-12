#THIS CODE RECORDS IMAGES AND KEYSTROKES WHILE PLAYING GAME
#RESULTS IN .npz FILE FOR TRAINING MODEL. 
import numpy as np
#import PIL
from PIL import ImageGrab
import cv2
import time
from sys import stdout
from IPython.display import clear_output
import threading
from heapq import nlargest
global grb
def screen_record(method = 'grabber'):

    rawIMG =  np.array(ImageGrab.grab(bbox=(0,40,1200,800)))
    generalIMG = cv2.resize(rawIMG, dsize=(240, 160), interpolation=cv2.INTER_CUBIC)
    #resizeIMG = cv2.resize(generalIMG, (1200,800))
    #cv2.imshow('window',cv2.cvtColor(resizeIMG, cv2.COLOR_BGR2RGB))
    #if cv2.waitKey(25) & 0xFF == ord('q'):
      #  cv2.destroyAllWindows()
    
    return generalIMG   

global front_buffer
global back_buffer
#front_buffer = np.zeros((1200, 800), dtype=np.int8)
#back_buffer = np.zeros((1200, 800), dtype=np.int8)
front_buffer = np.zeros((240,160), dtype=np.int8)
back_buffer = np.zeros((240,160), dtype=np.int8)


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
    proccessed_image = image
    #cv2.resize(image,(480,270))
    return proccessed_image

from game_control import PressKey, ReleaseKey
from getkeys import key_check

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [A,W,D] boolean values.
    '''
    output = [0,0,0,0] #Keystrokes are recorded in 4x1 array for WASD.
    
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
    
    # Frames per second capture rate
    capturerate = 20.0
    while True:
        last_time = time.time()
        if len(seq) == 5:
            del seq[0]

        seq.append(preprocess_image(np.copy(back_buffer)))
        num = num + 1
        keys = key_check()
        key_out = keys_to_output(keys)
        waittime = (1.0/capturerate)-(time.time()-last_time)
        if waittime>0.0:
            time.sleep(waittime)

def counter_keys(key):
        if np.array_equal(key , [0,0,0,0]):
            return 0
        elif np.array_equal(key , [1,0,0,0]):
            return 1
        elif np.array_equal(key , [0,1,0,0]):
            return 2
        elif np.array_equal(key , [0,0,1,0]):
            return 3
        elif np.array_equal(key , [0,0,0,1]):
            return 4
        elif np.array_equal(key , [1,0,1,0]):
            return 5
        elif np.array_equal(key , [1,0,0,1]):
            return 6
        elif np.array_equal(key , [0,1,1,0]):
            return 7
        elif np.array_equal(key , [0,1,0,1]):
            return 8
        else:
            return -1

def save_data(data,number):
    file_name = 'C:\\Users\\austi\\Dropbox\\ISU Semester 8\\ME 592 ML CPS\\Final\\Final_Project_592\\Final_Project_592\\DATA6\\Training_'+str(number)+'.npz'
    np.savez_compressed(file_name,data)
    del data




def run():
    global fps
    global front_buffer
    global back_buffer
    global seq
    global key_out
    global num
    global count
    training_data = []
    threads = list()
    th_img = threading.Thread(target=img_thread)
    th_seq = threading.Thread(target=image_sequencer_thread)
    threads.append(th_img)
    threads.append(th_seq)
    th_img.start()
    time.sleep(1)
    th_seq.start()
    l = 0
    fn = 0
    time.sleep(4)
    last_num = 0
    count = 0
    
    number_of_keys = [0,0,0,0,0,0,0,0,0]
    
    while True:
        img_seq = seq.copy()
        output = key_out.copy()
        
        while len(img_seq) != 5 or last_num==num:
            del img_seq, output
            img_seq = seq.copy()
            output = key_out.copy()
        last_num = num
        
        clear_output(wait=True)
        stdout.write('Recording at {} FPS \n'.format(fps))
        stdout.write('Images in sequence {} \n'.format(len(img_seq)))
        stdout.write('Training data len {} secuences \n'.format(l))
        stdout.write('Number of archives {}\n'.format(fn))
        stdout.write('Keys pressed: ' + str(output) + ' \n')
        stdout.write('Keys samples in this file: ' + 'none:' + str(number_of_keys[0]) + ' A:' + str(number_of_keys[1])+ ' D:' + str(number_of_keys[2]) + ' W:' + str(number_of_keys[3])+ ' S:' + str(number_of_keys[4]) + ' AW:'  + str(number_of_keys[5]) + ' AS:' + str(number_of_keys[6]) + ' WD:' + str(number_of_keys[7]) + ' SD:' + str(number_of_keys[8]) + ' \n')
        stdout.flush()
        
        key  = counter_keys(output)
        
        if key != -1:
            larg = nlargest(9,number_of_keys)
            prop = (9. - float(larg.index(number_of_keys[key])))/10
            if(number_of_keys[key]  > np.mean(number_of_keys) * 1.25):
                prop = prop + 0.05
            if (np.random.rand() > prop):
                number_of_keys[key] += 1
                l = l+1
                training_data.append([img_seq[0],img_seq[1],img_seq[2],img_seq[3],img_seq[4], output])
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if len(training_data) % 200 == 0:
            print(len(training_data))
            threading.Thread(target=save_data, args=(training_data.copy(), fn,)).start()
            fn = fn + 1
            del training_data
            training_data = []
            

def number_instances_per_class(data):
    
    nonekey = []
    A = []
    D = []
    W = []
    S = []
    
    AD = []
    AW = []
    AS = []
    DW =[]
    DS = []
    WS =[]
    
    ADW = []
    AWS =[]
    ADS = []
    DWS = []
    
    ASWS = []
    
    np.random.shuffle(data)
    
    for d in data:
        if np.array_equal(d[5] , [0,0,0,0]):
            nonekey.append(d)
        elif np.array_equal(d[5] , [1,0,0,0]):
            A.append(d)
        elif np.array_equal(d[5] , [0,1,0,0]):
            D.append(d)
        elif np.array_equal(d[5] , [0,0,1,0]):
            W.append(d)
        elif np.array_equal(d[5] , [0,0,0,1]):
            S.append(d)
        elif np.array_equal(d[5] , [1,1,0,0]):
            AD.append(d)
        elif np.array_equal(d[5] , [1,0,1,0]):
            AW.append(d)
        elif np.array_equal(d[5] , [1,0,0,1]):
            AS.append(d)
        elif np.array_equal(d[5] , [0,1,1,0]):
            DW.append(d)
        elif np.array_equal(d[5] , [0,1,0,1]):
            DS.append(d)
        elif np.array_equal(d[5] , [0,0,1,1]):
            WS.append(d)
        elif np.array_equal(d[5] , [1,1,1,0]):
            ADW.append(d)
        elif np.array_equal(d[5] , [1,1,0,1]):
            AWS.append(d)
        elif np.array_equal(d[5] , [1,1,0,1]):
            ADS.append(d)
        elif np.array_equal(d[5] , [0,1,1,1]):
            DWS.append(d)
        elif np.array_equal(d[5] , [1,1,1,1]):
            ASWS.append(d)
        return [nonekey,A,D,W,S,AW,AS,DW,DS]


def balance_data(data_in_clases):
    balanced_data = []
    data_in_clases.sort(key=len)
    max_len = len(data_in_clases[0])
        
    for data in data_in_clases:
        if len(data) > max_len:
            data=data[:max_len]
        for d in data:
            balanced_data.append(d)
        
    np.random.shuffle(balanced_data)
    
    return balanced_data    

#time.sleep(1)
run()

debug = False
if debug:
    with np.load('C:\\Users\\abe_mbteeple\\Desktop\\Class Material\\Spring 2022\\Training_Data\\Training_Data_1.npz', allow_pickle=True) as data:
        training_data = data['arr_0']
    number = number_instances_per_class(training_data)
    print('none: ' + str(len(number[0])))
    print('A: ' + str(len(number[1])))
    print('D ' + str(len(number[2])))
    print('W ' + str(len(number[3])))
    print('S ' + str(len(number[4])))
    print('AW ' + str(len(number[5])))
    print('AS ' + str(len(number[6])))
    print('DW ' + str(len(number[7])))
    print('DS ' + str(len(number[8])))

    balanced_data = balance_data(number)
    number = number_instances_per_class(balanced_data)
    print('none: ' + str(len(number[0])))
    print('A: ' + str(len(number[1])))
    print('D ' + str(len(number[2])))
    print('W ' + str(len(number[3])))
    print('S ' + str(len(number[4])))
    print('AW ' + str(len(number[5])))
    print('AS ' + str(len(number[6])))
    print('DW ' + str(len(number[7])))
    print('DS ' + str(len(number[8])))


