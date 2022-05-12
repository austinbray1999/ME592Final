import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directkeys import PressKey, ReleaseKey, W, A, S, D

#Used to press keys in GTA based on whether you want to go straight, left, right, or neutral
def straight():
  PressKey(W)
  ReleaseKey(A)
  ReleaseKey(D)

def left():
  PressKey(A)
  ReleaseKey(W)
  ReleaseKey(D)
  ReleaseKey(A)

def right():
  PressKey(D)
  ReleaseKey(A)
  ReleaseKey(W)
  ReleaseKey(D)

def slow_ya_roll():
  ReleaseKey(W)
  ReleaseKey(A)
  ReleaseKey(D)

img = cv2.imread("C:\\Users\\Jade\\Documents\\LaneSet\\C25_5_w.png")

#Convert pic to gray scale and detect edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

#Use probabilistic Hough Lines to detect extremes with the minimum line length of 45
#and the maximum gap between lines of 2
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=45, maxLineGap=2)

pts = np.array([[190, 350], [590, 230], [230,230], [790, 350]])
coord_pts = []

#Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

    #Determine if lines are within the bounding box points
    if (x1 >= pts[0,0]) and (x1 <= pts[1,0]) and (x2 >= pts[2,0]) and (x2 <= pts[3,0]):
        lanepts = x1, y1, x2, y2
        coord_pts.append(lanepts)
        x_coords = (lanepts[0], lanepts[2])
        y_coords = (lanepts[1], lanepts[3])
        A = vstack([x_coords, ones(len(x_coords))]).T
    #Turn right if we detect the left line, turn left if we detect the right line, otherwise go straight
    if x1 <= pts[0,0]:
        right()
    if x2 >= pts[1,0]:
        left()
    else:
        straight()

#Plot image with bounding box points
plt.imshow(img)
plt.scatter(pts[:,0], pts[:,1], marker="x", color="blue", s=200)
plt.show()







