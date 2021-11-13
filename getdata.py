import os, random, shutil

data_dir = './data'
train_dir = './dataset/train'
test_dir = './dataset/test'
train_ratio = 0.8
val_ratio = 0.2

for files in os.listdir(data_dir):
    image_name = []
    train_path = train_dir + '/' + files
    test_path = test_dir + '/' + files
    print(len(os.listdir(train_path)))
    print(len(os.listdir(test_path)))

#
# for files in os.listdir(data_dir):
#     image_name = []
#     train_path = train_dir + '/' + files
#     test_path = test_dir + '/' + files
#     # os.mkdir(train_path)
#     # os.mkdir(test_path)
#     class_path = data_dir + '/' +files
#     filenum = len(os.listdir(class_path))
#     # print(filenames)
#     num_train = int(filenum * train_ratio)
#     for file in os.listdir(class_path):
#         image_name.append(file)
#     sample_train = random.sample(image_name, num_train)
#     for name in sample_train:
#         shutil.move(os.path.join(class_path, name), train_path)
#
#     sample_val = list(set(image_name).difference(set(sample_train)))
#
#     for name in sample_val:
#         shutil.move(os.path.join(class_path, name), test_path)