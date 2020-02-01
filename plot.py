import os
import glob
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')

least = 100
least_file = ''
max_pre_file = ''
max_pre = 0
log_files = glob.glob("metrics/*.txt")
log_files.sort(key=lambda x: os.path.getmtime(x))
loss_array = []
valid_loss_array = []
recall_array = []
precision_array = []
counter = 0
for f in log_files:
    with open(f, 'r') as file:
        print(f)
        all_lines_list = file.readlines()
    for line in all_lines_list:
        if "Epoch loss" in line:
            counter += 1
            loss_line = line
            loss_line = loss_line[13:]
            print("Loss: {}".format(loss_line))
            loss_line = float(loss_line)
            loss_array.append(loss_line)
        elif "Validation Loss" in line:
            val_line = line
            val_line = val_line[18:]
            print("Validation Loss: {}".format(val_line))
            val_line = float(val_line)
            valid_loss_array.append(val_line)
            if val_line < least:
                least = val_line
                least_file = f
        elif "Recall" in line:
            val_line = line
            val_line = val_line[9:]
            print("Recall: {}".format(val_line))
            val_line = float(val_line)
            recall_array.append(val_line)
        elif "Precision" in line:
            val_line = line
            val_line = val_line[12:]
            print("Precision: {}".format(val_line))
            val_line = float(val_line)
            if val_line > max_pre:
                max_pre = val_line
                max_pre_file = f
            precision_array.append(val_line)
        else:
            pass
    

print("-----------------------------------------------------------------------\n{}\n{}".format(least, least_file))
epochs = range(0, counter)
plt.plot(epochs, loss_array, 'g' , label='Training loss')
plt.plot(epochs, valid_loss_array, 'b' , label='Validation loss')
#plt.plot(epochs, recall_array, 'r' , label='Recall')
#plt.plot(epochs, precision_array, 'y' , label='Precision')
plt.title('YoloV3 - Object detection')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.figtext(.2, .97, "least val loss: {} - {}".format(least, least_file))
plt.legend()
plt.savefig('metrics/loss.png')
