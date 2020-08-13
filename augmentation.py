import random
from PIL import Image

# path = 'ISIC_2019_Training_GroundTruth_train_0.2.csv'
img_dir = 'ISIC_2019_Training_Input/' 
path = 'ISIC_2019_Training_GroundTruth_sub1_train.csv'
augment_fraction = 0.5

fn = open(path, 'r+')
data = []
for line in fn:
    data.append(line)
fn.close

len_data = len(data)
offset = int(len_data * augment_fraction)

change_data = data[:offset]
augment_data = []

for line in change_data:
    line = line.rstrip()
    name = line.split(',')[0]
    label = line.split(',')[1:]
    label = ",".join(label)

    img = Image.open(img_dir+name+'.jpg')
    out = img.transpose(Image.ROTATE_90)
    new_name = name + '_rotated'
    out.save(img_dir+new_name+'.jpg')

    new_line = new_name + ',' + label + '\n'
    augment_data.append(new_line)

for index in range(offset):
    data.append(augment_data[index])

random.shuffle(data)

path_augment = 'ISIC_2019_Training_GroundTruth_sub1_train_aug' + str(augment_fraction) + '.csv'
fa = open(path_augment, 'w')

for line in data:
    fa.write(line)
fa.close
