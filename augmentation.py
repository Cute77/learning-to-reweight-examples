import random
from PIL import Image

# path = 'ISIC_2019_Training_GroundTruth_train_0.2.csv'
img_dir = 'ISIC_2019_Training_Input/' 
path = 'ISIC_2019_Training_GroundTruth_sub1_train.csv'
augment_fraction = 0.75

fn = open(path, 'r+')
data = []
datas = []
for line in fn:
    data.append(line)
fn.close

len_data = len(data)
offset = int(len_data * (augment_fraction-0.5))
de = int(len_data * 0.5)

change_data = data[de: (de+offset)]
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

patha = 'ISIC_2019_Training_GroundTruth_sub1_train_aug0.5.csv'
faa = open(patha, 'r+')
for line in faa:
    datas.append(line)

for index in range(offset):
    datas.append(augment_data[index])

# random.shuffle(data)

path_augment = 'ISIC_2019_Training_GroundTruth_sub1_train_aug' + str(augment_fraction) + '.csv'
fa = open(path_augment, 'w')

for line in datas:
    fa.write(line)
fa.close
