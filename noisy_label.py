import random

# path = 'ISIC_2019_Training_GroundTruth_train_0.2.csv'
path = 'ISIC_2019_Training_GroundTruth_sub5000_train.csv'
noise_fraction = 0.4

fn = open(path, 'r+')
data = []
for line in fn:
    data.append(line)
fn.close


# print(data)
len_data = len(data)
offset = int(len_data * noise_fraction)

change_data = data[:offset]
noisy_data = []
i = 1

for line in change_data:
    i = i + 1
    line = line.rstrip()
    name = line.split(',')[0] + ','
    label = line.split(',')[1:]
    temp = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # print('id: ', id)
    # print('temp: ', temp)
    # print('label: ', label)

    if '1.0' in line:
        temp.pop(label.index('1.0'))
        random.shuffle(temp)

        id = label.index('1.0')
        label[label.index('1.0')] = '0.0'
        label[temp[0]] = '1.0'

    else:
        print(i)
        random.shuffle(temp)
        print('name: ', name)
        print('label: ', label)
        print(temp[0], '\n')
        label[temp[0]] = '1.0'
        
    label = ",".join(label)
    # marked by "1": noisy label, changed label
    line = "1 " + str(id) + " " + name + label + '\n'
    noisy_data.append(line)


datas = []
for index in range(offset):
    datas.append(noisy_data[index])


for index in range(offset, len(data)):
    # marked by "0": clean label, unchanged label
    line = "0 0 " + data[index]
    datas.append(line)

random.shuffle(datas)

pathtwo = 'ISIC_2019_Training_GroundTruth_markedgiven5000_train_' + str(noise_fraction) + '.csv'
ftwo = open(pathtwo, 'w')

for line in datas:
    ftwo.write(line)
ftwo.close

