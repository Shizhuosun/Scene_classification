import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import os

# import os
#
# train_path = './data'
#
# for files in os.listdir(train_path):
#     train_path2 = os.path.join(train_path, files)
#     print(len(os.listdir(train_path2)))


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")


    else:
        print("---  There is this folder!  ---")


def data_enhancement(img_input_path,img_output_path):
    image = load_img(img_input_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    img_dag = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range = 0.2,
                                 shear_range = 0.3,
                                 zoom_range = 0.3,
                                 horizontal_flip = True,
                                 fill_mode="nearest")

    img_generator = img_dag.flow(image, batch_size=1,
                                 save_to_dir=img_output_path,
                                 save_prefix = "image", save_format = "jpg")
    count =0 #计数器
    for img in img_generator:
        count += 1
        if count == 15:  #生成多少个样本后退出
            break

if __name__=="__main__":
    image_input_dir = './train'
    image_out_dir = "./data"
    #print(os.listdir(image_input_dir))
    for file in os.listdir(image_input_dir):
        class_image_path = os.listdir(image_input_dir + "/" + file)
        # print(file)
        image_out_path = image_out_dir + '/' + file
        mkdir(image_out_path)
        for files in class_image_path:
            # print(image_input_dir + '/' + file + '/' + files)
            image_path = image_input_dir + '/' + file + '/' + files
            data_enhancement(image_path, image_out_path)
    # date_enhancement(image_path, image_out_path)