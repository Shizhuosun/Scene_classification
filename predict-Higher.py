import json
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from resnest.torch import resnest50
import torch
from PIL import Image
from torchvision import transforms


num_classes = 15
N_worker = 0 # GPU 4, CPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load_model_path = '../weight_model/resnest50-528c19ca.pth'
load_model_path = './model/resnest50_new.pth'
# configure on cloud GPU (train&test)
# save_path = './resnest50_new.pth'
data_dir = '../dataset'
test_dir = '../dataset/test'

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes=15, feature_extract=True,
                     model_name='resnest50', use_pretrained=True):
    model_ft = None  # 用于提取特征的预训练模型
    input_size = 0

    if model_name == 'resnest50':
        # torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        # # load pretrained models, using ResNeSt-50 as an example
        # model_ft = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        # model.eval()
        model_ft = resnest50(pretrained=use_pretrained)
        # 冻结原参数
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features  # 获取输出层的输入维度
        # 修改输出层的shape(原为2048in1000out)
        # 最后一层的名字可以通过print模型来查看
        # resnest中最后一层为'fc'

        # CHOOSE ONE OF THE CHANGES
        # change 1
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # change 2
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, num_classes))

        # crop size
        input_size = 224

    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    # print("Params to learn:")
    # if feature_extract:
    #     params_to_update = []
    #     for name, param in model_ft.named_parameters():
    #         if param.requires_grad == True:
    #             params_to_update.append(param)
    #             print("\t", name)
    # else:  # train from scratch
    #     for name, param in model_ft.named_parameters():
    #         if param.requires_grad == True:
    #             print("\t", name)

    optimizer_ft = optim.Adam(params_to_update, lr=0.001)

    return model_ft, optimizer_ft, input_size

def online_predict(model, input_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read class_indict
    json_path = './model/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    # load image
    img_path = 'C:\\Users\\think\\Desktop\\5.jpg'
    # img_path = "./dataset/test/Coast/image_0_125.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()



def main():
    print('Predict...')
    model_ft, _, input_size = initialize_model()
    ######## Note: if using CPU, must declare map_location
    model_ft.load_state_dict(torch.load(load_model_path, map_location=device))
    online_predict(model_ft, input_size)

if __name__ == '__main__':
    main()