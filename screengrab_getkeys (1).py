import pyautogui 
import keyboard

i = 1
while True:
    try:
        while i < 10:
            if keyboard.is_pressed('w'):
                hwin = pyautogui.screenshot()
                hwin.save(str(i) + '_w.png')
                i = i + 1
            elif keyboard.is_pressed('a'):
                hwin = pyautogui.screenshot()
                hwin.save(str(i) + '_a.png')
                i = i + 1
            elif keyboard.is_pressed('s'):
                hwin = pyautogui.screenshot()
                hwin.save(str(i) + '_s.png')
                i = i + 1
            elif keyboard.is_pressed('d'):
                hwin = pyautogui.screenshot()
                hwin.save(str(i) + '_d.png')
                i = i + 1
            elif keyboard.is_pressed('f'):
                hwin = pyautogui.screenshot()
                hwin.save(str(i) + '_f.png')
            if i == 10:
                print("You've reached the limit. Exit kernel")
                break
    except:
        break
        

    
