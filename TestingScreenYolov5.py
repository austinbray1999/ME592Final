import torch
from PIL import ImageGrab
# pip install pyyaml

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Image
img = ImageGrab.grab(bbox=(500, 500, 1600, 1700))  # take a screenshot

# Inference
results = model(img)
results.show()