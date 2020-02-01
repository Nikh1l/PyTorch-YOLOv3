
with open('../data/custom_data/train.txt', 'r') as f:
    filelist = f.readlines()

files = []
for file in filelist:
    file = file.replace("train", "images")
    files.append(file)

with open('../data/custom_data/train.txt', 'w') as f:
    for file in files:
        f.write(file)

with open('../data/custom_data/test.txt', 'r') as f:
    filelist = f.readlines()

files = []
for file in filelist:
    file = file.replace("test", "images")
    files.append(file)

with open('../data/custom_data/test1.txt', 'w') as f:
    for file in files:
        f.write(file)
