'''
Please read first!
In the training process, I spilt the train dataset into train and validation dataset, so the parameter "train_data_dir"
in the def train() refers to the data_dir ,in the data_dir, there are training dataset and validation dataset respectively.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from PIL import Image
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import os
import copy
from resnest.torch import resnest50


num_classes = 15
N_worker = 0 # GPU 4, CPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load_model_path = '../weight_model/resnest50-528c19ca.pth'
# load_model_path = './resnest50_new.pth'
# configure on cloud GPU (train&test)
# save_path = './resnest50_new.pth'
# data_dir = '../dataset'
# test_dir = '../dataset/test'
test_name_label_pair_path = './test_name_label.txt'

# functions & classes
def train_model(model, dataloaders, criterion, optimizer,num_epochs):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        for phase in ['train', 'test']:
            # 根据phase设置mode(主要是在dropout等结构上的区别)
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 每个batch进行梯度清零
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 将验证准确率最高的模型参数做副本拷贝
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 重新加载回验证准确率最高的模型参数,并返回模型
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, best_acc


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

class Test_dataset(Dataset):
    def __init__(self, data_path, pair_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.pairs = []
        self.mapping_n2o = {"Coast": 0, "Forest": 1, "Highway": 2, "Insidecity": 3, "Mountain": 4, "Office": 5, "OpenCountry": 6, "Street": 7, "Suburb": 8, "TallBuilding": 9,
                                "bedroom": 10, "industrial": 11, "kitchen": 12, "livingroom": 13,"store": 14}
        with open(pair_path, 'r', encoding='utf-8') as f:
            for line in f:
                ld = line.strip()
                if ld:
                    ll = ld.split(', ')
                    lt = tuple([ll[0], self.mapping_n2o[int(ll[1])]])
                    self.pairs.append(lt)
        print('Num test samples:', len(self.pairs))
        print()

    def __len__(self):
        nl = os.listdir(self.data_path)
        return len(nl)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.pairs[idx]
        photo_path = os.path.join(self.data_path, pair[0])
        label = np.array(pair[1])  # .astype('float')
        # 以灰度图像的形式读取图片
        image = Image.open(photo_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        sample = {'label': label, 'image': image}
        return sample


def test_model(model, input_size, test_dir):
    batch_size = 16
    total = 0.0
    correct = 0.0
    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # testset =Test_dataset(test_dir, name_label_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=N_worker)
    model.eval()  # 验证时关闭BN和Dropout
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            input, target = batch
            output = model(input)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += torch.sum(predicted == target.data)
            if i % 100 == 99:
                print(i)
                print('batch: %5d,\t acc: %f' % (i + 1, correct / total))
    print('--test accu: {}'.format(100.0 * correct / len(testloader.dataset)))
    return 100.0 * correct / total


def train(train_data_dir, model_dir):
    num_epochs = 30
    batch_size = 64
    print('Train...')
    model_ft, optimizer_ft, input_size = initialize_model()
    # Create training and validation datasets
    criterion = nn.CrossEntropyLoss()
    data_transforms = {
        # Data augmentation and normalization for training
        'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Just normalization for validation
        'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(train_data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                                           num_workers=N_worker) for x in ['train', 'test']}
    # train & val
    model_ft, hist, best_acc = train_model(model_ft, dataloaders_dict, criterion,
                                     optimizer_ft, num_epochs)
    # save
    torch.save(model_ft.state_dict(), model_dir)
    print('model saved.')
    return best_acc

def test(test_dir, load_model_path):  # 3900
    print('Test...')
    model_ft, _, input_size = initialize_model()
    # load parameters(only, no structure)
    ######## Note: if using CPU, must declare map_location
    model_ft.load_state_dict(torch.load(load_model_path, map_location=device))
    print('checkpoint loaded.')
    test_model(model_ft, input_size, test_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='test', choices=['train', 'test'])
    parser.add_argument('--train_data_dir', default='./dataset/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./dataset/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='./model/resnest50_new.pth', help='the pre-trained model')
    opt = parser.parse_args()


    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)
