import random
# import os

path = 'ISIC_2019_Training_GroundTruth_noisy.csv'

path_train = 'ISIC_2019_Training_GroundTruth_train.csv'
path_test = 'ISIC_2019_Training_GroundTruth_test.csv'

# os.mknod(path_train)
# os.mknod(path_test)

fn = open(path, 'r+')
data = []
for line in fn:
    data.append(line)

random.shuffle(data)
len_data = len(data)
offset = int(len_data * 0.8)

fm = open(path_train, 'w')
fh = open(path_test, 'w')
train_data = data[:offset]
test_data = data[offset:]

for line in train_data:
    fm.write(line)
fm.close

for line in test_data:
    fh.write(line)
fh.close




