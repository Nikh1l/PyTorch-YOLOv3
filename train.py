from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from evaluate import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import glob
import re


import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim



# get the latest checkpoint
last_weight = 0
latest_file = None
try:
    list_of_files = glob.glob('checkpoint/*.weights') # * means all if need specific format then *.ext
    latest_file = max(list_of_files, key=os.path.getctime)
    last_weight = int(re.findall(r'\d+', latest_file)[1])
    last_weight += 1
except Exception as e:
    print(e)
    last_weight = 0

print(last_weight)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="config/yolov3-tiny.weights", help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)


    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            if latest_file:
                print("Using weights: ", latest_file)
                model.load_darknet_weights(latest_file)
            else:    
                model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            if latest_file:
                print("Using weights: ", latest_file)
                model.load_darknet_weights(latest_file)
            else:
                model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    epoch_loss = []
    for epoch in range(opt.epochs):
        acc_loss = 0.0
        mAP_epoch = ''
        loss_epoch = ''
        model.train()
        valid_loss = 0
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]


            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            acc_loss += loss.item()
            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)
            

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class, validation_loss, confusion_log = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.4,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            ap_table_str = str(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            mAP_epoch = f"---- mAP {AP.mean()}"
            print(f"---- Validation Loss {validation_loss}")
            print(confusion_log)
            valid_loss = validation_loss
            f1_score = f1.mean()


        if epoch % opt.checkpoint_interval == 0:
            epoch_loss = acc_loss/len(dataloader)
            torch.save(model.state_dict(), f"checkpoint/yolov3-tiny_%d.pth" % (last_weight + epoch))
            model.save_darknet_weights(f"checkpoint/yolov3-tiny_%d.weights" % (last_weight + epoch))
            path = f"metrics/yolov3-tiny_%d.txt" % (last_weight + epoch)
            with open(path, 'w') as f:
                f.write("Epoch loss : {}\n".format(epoch_loss))
                f.write("Recall : {}\n".format(recall.mean()))
                f.write("Precision : {}\n".format(precision.mean()))
                f.write("F1 Score : {}\n".format(f1_score))
                f.write("Validation Loss : {}\n".format(valid_loss))
                f.write("{}\n".format(ap_table_str))
                f.write("{}\n".format(mAP_epoch))
                f.write("Eval Statistics: \n{}".format(confusion_log))
