'''
resize 256*256

'''
import cv2
import os

train_path = './data'

list1=[]
for file in os.listdir(train_path):
    class_image_path = os.listdir(train_path + "/" + file)
    # print(file)
    image_out_path = train_path + '/' + file
    for files in class_image_path:
        # print(image_input_dir + '/' + file + '/' + files)
        image_path = train_path + '/' + file + '/' + files
        img = cv2.imread(image_path)
        dst_size = (256, 256)
        img_resize = cv2.resize(img, dst_size, interpolation = cv2.INTER_AREA)
        cv2.imwrite(image_path, img_resize)
        print(img_resize.shape)

# for files in os.listdir(train_path):
#     train_path2 = os.listdir(train_path + "/" + files)
#     for file in train_path2:
#         train_path = train_path + '/' + file + '/' + files
#         print(train_path+ '/n')
#         img = cv2.imread(train_path)
#         dst_size = (256, 256)
#         img_resize = cv2.resize(img, dst_size, interpolation = cv2.INTER_AREA)
#         cv2.imwrite(train_path, img_resize)
#         print(img_resize.shape)
