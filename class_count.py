import os
import glob
from terminaltables import AsciiTable


train_path = "data/custom_data/train.txt"
test_path = "data/custom_data/test.txt"


with open(train_path, 'r') as f:
    train_files = f.readlines()
    
train_list = []
for file in train_files:
    file = file.replace("images", "labels").replace(".jpg", ".txt")
    file = file[:-1]
    if file[-2:] == "tx":
        file = file.replace("tx", "txt")
    train_list.append(file)
print(train_list)

with open(test_path, 'r') as f:
    test_files = f.readlines()

test_list = []
for file in test_files:
    file = file.replace("images", "labels").replace(".jpg", ".txt")
    file = file[:-1]
    if file[-2:] == "tx":
        file = file.replace("tx", "txt")
    test_list.append(file)
print(test_list)

    
train_cell_counter = 0
train_laptop_counter = 0
log_str = ""
for file in train_list:
    with open(file, 'r') as f:
        lines = f.readlines()
        #print("Reading file : {}".format(file))
        #print("Number of lines: {}".format(len(lines)))
        temp_lap = 0
        temp_cell = 0
        for line in lines:
            if line != "":
                if line[0] == "0":
                    train_cell_counter += 1
                    temp_cell += 1
                elif line[0] == "1":
                    train_laptop_counter += 1
                    temp_lap += 1
        if temp_cell >= 2:
            print("Reading file : {}".format(file))
            print("Training:\nNumber of cell phones = {} \nNumber of laptops = {}".format(temp_cell, temp_lap))

test_cell_counter = 0
test_laptop_counter = 0
for file in test_list:
    with open(file, 'r') as f:
        lines = f.readlines()
        #print("Reading file : {}".format(file))
        #print("Number of lines: {}".format(len(lines)))
        temp_lap = 0
        temp_cell = 0
        for line in lines:
            if line != "":
                if line[0] == "0":
                    test_cell_counter += 1
                    temp_cell += 1
                elif line[0] == "1":
                    test_laptop_counter += 1
                    temp_lap += 1
        #print("Testing:\nNumber of cell phones = {} \nNumber of laptops = {}".format(temp_cell, temp_lap))

print("-------------------------------------------------------------------------------------")
print("Final Table")
print("Training: Cellphone={}, Laptop={}".format(train_cell_counter, train_laptop_counter))
print("Testing: Cellphone={}, Laptop={}".format(test_cell_counter, test_laptop_counter))
final_table = [["", "Cell phone", "Laptop"], ["Training", train_cell_counter, train_laptop_counter]]
final_table += [["", " ", " "], ["Testing", test_cell_counter, test_laptop_counter]]
print(AsciiTable(final_table).table)
log_str += AsciiTable(final_table).table
with open("class_count_table.txt", 'w') as f:
    f.write(log_str)