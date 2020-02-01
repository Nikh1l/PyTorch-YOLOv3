from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator 

# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    loss_list = []
    mci_list = []
    for batch_i, (img_paths, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        images = Variable(imgs.to(device))
        tars = Variable(targets.to(device), requires_grad=False)
        
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
                
        annotations = targets[targets[:, 0] == batch_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        target_boxes = annotations[:, 1:]

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            ls, ot = model(images, tars)
            loss_list.append(ls.item())

        batch_metrics, miss_classified_images = get_batch_statistics(outputs, targets, iou_threshold=iou_thres, image_paths=img_paths)
        sample_metrics += batch_metrics
        mci_list += miss_classified_images

        #----------------------------------------------------------------------------#
        '''
        Running inference and eval at the last epoch.

        Todo: Segregate True positive, False positive and False negative images basec on (1. whole image  2. individual object). 
        Compare with ground truth boxes and iou thres.
        detection_pi : Detections per image
        gt_pi : Ground truth values per image 
        labels : Labels per images
        path : Image path
        '''

        # Create folders and subfolders

        main_folder = f"qualitative_output/{epoch}"
        fp_folder = f"{main_folder}/fp_images"
        tp_folder = f"{main_folder}/tp_images"
        fn_images = f"{main_folder}/ fn_images"

        os.makedirs(main_folder, exist_ok=True)
        os.makedirs(fn_images, exist_ok=True)
        os.makedirs(fp_folder, exist_ok=True)
        os.makedirs(tp_folder, exist_ok=True)

        
        for index, (detection_pi, gt_pi, labels, path) in enumerate(zip(outputs, target_boxes, target_labels, img_paths)):
            
            num_detection = len(detection_pi)
            num_target = len(target_boxes)

            if num_detection > num_target:
                #Fasle positive
                save_folder = fp_folder
            elif num_detection == num_target:
                if path in mci_list:
                    # Miss classified image
                    # iou < IOU tresh 
                    save_folder = fp_folder
                else:
                    # True positive image
                    save_folder = tp_folder
            elif num_detection < num_target:
                # False negative image
                save_folder = fn_images
            
            
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            gt_pi = rescale_boxes(gt_pi, img_size, img.shape[:2])

            if detection_pi is not None:
                detection_pi = rescale_boxes(detection_pi, img_size, img.shape[:2])
                unique_labels = detection_pi[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection_pi:
                    # Todo: draw bounding boxes 

                for (x1, y1, x2, y2), target_class in zip(gt_pi, labels):
                    # Todo: draw ground truth bounding boxes
        
        # Save the image in respective folders
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"{save_folder}/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
            
        #----------------------------------------------------------------------------#

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class, confusion_log = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    validation_loss = sum(loss_list)/len(dataloader)
    return precision, recall, AP, f1, ap_class, validation_loss, confusion_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="config/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class, validation_loss, confusion_log = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
    print(f"precision: {precision}")    
    print(f"recall: {recall}")    
    print(f"f1 score: {f1}")    
