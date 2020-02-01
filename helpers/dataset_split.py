import glob
import os
from random import shuffle

print("Reading file...")
with open("data/custom_dataset/images.txt") as f:
    files = f.readlines()

shuffle(files)
print("Shuffling entries...")

n_files = len(files)
ten_per = int(20*n_files/100)
print("Number of entries: %d" %n_files)
print("20 percent of entries: %d" %ten_per)

val_list = []
train_list = []

print("Splitting train and test set...")
for index, item in enumerate(files):
    if index+1 < ten_per:
        val_list.append(item)
    else:
        train_list.append(item)

print("Creating files train.txt and val.txt")
with open("data/custom_dataset/train.txt", "w") as f:
    for item in train_list:
        f.write("%s"%item)

with open("data/custom_dataset/val.txt", "w") as f:
    for item in val_list:
        f.write("%s"%item)