from hubconf import custom
from models.yolo import Model
from utils.torch_utils import select_device
import torch

model = custom("C:/Development/Robotics/FRC/Test_Vision/ML/yolov7/stuff/best.pt")

# test model with image in stuff folder
# result = model.predict("C:/Development/Robotics/FRC/Test_Vision/ML/yolov7/stuff/img.jpg")

hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
hub_model.load_state_dict(model.float().state_dict())  # load state_dict
hub_model.names = model.names  # class names
hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
thing = hub_model.to(device)

import cv2

img = cv2.imread("C:/Development/Robotics/FRC/Test_Vision/ML/yolov7/stuff/img.jpg")
img = cv2.resize(img, (540, 960))

results = thing(img)  # batched inference
results.print()
results.save()