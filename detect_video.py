from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2
from tqdm import tqdm

from PIL import Image
#from moviepy.editor import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
plt.switch_backend('agg')


def detect_image(img):
    # scale and pad image

    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0), 
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    detections = detections[0]
    img = np.array(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    print(detections)
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            cls = classes[int(cls_pred)]
            cv2.rectangle(img, (x1, y1), (x1+box_w, y1+box_h),color, 4)
            cv2.rectangle(img, (x1, y1-35), (x1+len(cls)*19+60,y1), color, -1)
            conf_float = float(conf.item())
            conf_float = round(conf_float, 2)
            cv2.putText(img, cls + "-" + str(conf_float) , (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--video_file", type=str, default="data/custom_data/test_vid/vid05.mp4", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3-tiny.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="config/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.80, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CUDA = torch.cuda.is_available()

    print("Loading network...")
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    
    model.eval()  # Set in evaluation mode
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    num_classes = 2
    img_size = opt.img_size
    
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    vid = cv2.VideoCapture(0)
    

    while True:
        ret, frame = vid.read()
        if ret == True:
            # cv2.imshow('frame',frame)
            pilimg = Image.fromarray(frame)
            img = detect_image(pilimg)
            cv2.imshow("Video", img)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break
    vid.release()
    cv2.destroyAllWindows()