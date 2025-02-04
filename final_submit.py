'''
Please read first!
In the training process, I spilt the train dataset into train and validation dataset, so the parameter "train_data_dir"
in the def train() refers to the data_dir ,in the data_dir, there are training dataset and validation dataset respectively.

how to run this code:

1. replace the path of the dataset(train data includes two dictionaries: train data and validation data)
   test data is a separate dictionary
2. load the model we had trained
    1) first, ensemble.pth is the final model which can be used to predict
    2) when load the ensemble.pth, it is necessary to load VGG.pth and RES.pth in the dictionary "ensemble"
    3) there may be different trained by GPU and CPU, so plz care about the main function
'''


import torch.optim as optim

from torchvision import datasets, models, transforms

from torch.utils.data import Dataset, DataLoader
import time
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(Residual_block, self).__init__()
        self.essential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = shortcut

    def forward(self, x):
        out = self.essential(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.relu(out)


def make_layer(in_channels, out_channels, num_block, stride=1):

    layers = []
    if stride != 1:
        # dotted line skip, indicating the increase of dimension/channel
        shortcut = nn.Sequential(
            # k=1,
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
    else:
        # solid line skip
        shortcut = None
    layers.append(Residual_block(in_channels, out_channels, stride, shortcut))
    # solid line skip
    for i in range(1, num_block):
        layers.append(Residual_block(out_channels, out_channels))

    return nn.Sequential(*layers)


class Resnet(nn.Module):
    def __init__(self, in_channel, num_labels=3):
        super(Resnet, self).__init__()
        self.pre = nn.Sequential(
            # (224+2p-k)//2 + 1 = c, k=7, c=112, so p=3
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # process the first maxpool separately, then it's convenient to divide the residual block
            # (112+2p-k)//2 + 1 = c, k=3, c=56, so p=1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 56*56*64 -> 56*56*64, s=1
        self.layer1 = make_layer(64, 64, num_block=3)
        # 56*56*64 -> 28*28*128, s=2
        self.layer2 = make_layer(64, 128, num_block=4, stride=2)
        # 28*28*128 -> 14*14*256, s=2
        self.layer3 = make_layer(128, 256, num_block=6, stride=2)
        # 14*14*256 -> 7*7*512, s=2
        self.layer4 = make_layer(256, 512, num_block=3, stride=2)

        # dense
        self.fc = nn.Linear(512, num_labels)

    def forward(self, x):
        xx = self.pre(x)

        xx = self.layer1(xx)
        xx = self.layer2(xx)
        xx = self.layer3(xx)
        xx = self.layer4(xx)

        # pool = nn.AvgPool2d(kernel_size=7)
        # xx = pool(xx)
        xx = F.avg_pool2d(xx, kernel_size=7)

        # print(xx.shape)
        xx = xx.view(xx.shape[0], -1)
        # print(xx.shape)

        return self.fc(xx)


class Retrained_res50(nn.Module):
    def __init__(self, device=device, para=None, ):
        super(Retrained_res50, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        # modify the input
        # print(resnet50.conv1) Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        pretrained_dict = resnet50.state_dict()  # 'conv1.weight'
        resnet50.load_state_dict(pretrained_dict)
        # modify the last FC
        fc_inputs = resnet50.fc.in_features  # 2048
        resnet50.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 15),
            # nn.LogSoftmax(dim=1)
        )
        if para:
            resnet50.load_state_dict(torch.load(para, map_location=device))
            for param in resnet50.parameters():
                param.requires_grad = False
        self.model = resnet50.to(device)

    def forward(self, x):  # [batch_size, channel:1, height, width]
        return self.model(x)


class Retrained_vgg(nn.Module):
    def __init__(self, device=device, para=None):
        super(Retrained_vgg, self).__init__()
        model_ft = models.vgg.vgg16(pretrained=False)
        pretrained_dict = model_ft.state_dict()
        model_ft.load_state_dict(pretrained_dict)
        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 15),
        )
        if para:
            model_ft.load_state_dict(torch.load(para, map_location=device))
            for param in model_ft.parameters():
                param.requires_grad = False
        self.model = model_ft.to(device)

    def forward(self, x):  # [batch_size, channel:1, height, width]
        return self.model(x)


def load_VGG(para, device=device):
    model_ft = models.vgg.vgg16(pretrained=False)
    retrained_dict = model_ft.state_dict()
    model_ft.load_state_dict(retrained_dict)
    model_ft.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 15),
    )
    if para:
        model_ft.load_state_dict(torch.load(para, map_location=device))
        for param in model_ft.parameters():
            param.requires_grad = False
    model_ft.to(device)
    return model_ft


def load_RES(para, device=device):
    resnet50 = models.resnet50(pretrained=False)
    # modify the input
    # print(resnet50.conv1) Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    retrained_dict = resnet50.state_dict()  # 'conv1.weight'
    resnet50.load_state_dict(retrained_dict)
    fc_inputs = resnet50.fc.in_features  # 2048
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 15),
        # nn.LogSoftmax(dim=1)
    )
    if para:
        resnet50.load_state_dict(torch.load(para, map_location=device))
        for param in resnet50.parameters():
            param.requires_grad = False
    return resnet50.to(device)


class Ensemble(nn.Module):
    def __init__(self, paras, num_classes, dnn_param=None, device=device):
        """
        build an ensemble model based on ResNet-50 and vgg.
        :param paras: [parameter path of ResNet, of vgg]
        :param device:
        :param dnn_param: parameter path of final DNN
        """
        super(Ensemble, self).__init__()
        self.res = load_RES(paras[0], device)
        self.res.fc = self.res.fc[:-2]  # output 256
        self.vgg = load_VGG(paras[1], device)

        self.vgg.classifier[6] = self.vgg.classifier[6][:-2]  # output 256
        dnn = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
        if dnn_param:
            # dnn.load_state_dict(torch.load('./dataset/xray_dnn_parameters.pkl'))
            dnn.load_state_dict(torch.load(dnn_param, map_location=device))
            for param in dnn.parameters():
                param.requires_grad = False
        self.dnn = dnn.to(device)

    def forward(self, x):  # [batch_size, channel:3, height, width]
        res_out = self.res(x)
        vgg_out = self.vgg(x)
        inputs = torch.cat((res_out, vgg_out), 1)
        return self.dnn(inputs)


num_classes = 15
N_worker = 0  # GPU 4, CPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load_model_path = '../weight_model/resnest50-528c19ca.pth'
# load_model_path = './resnest50_new.pth'
# configure on cloud GPU (train&test)
# save_path = './resnest50_new.pth'
# data_dir = '../dataset'
# test_dir = '../dataset/test'
test_name_label_pair_path = './test_name_label.txt'


# functions & classes
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
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
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
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


    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# load the model VGG.pth and RES.pth
def initialize_model(num_classes=15, feature_extract=True,
                     model_name='DNN', use_retrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'DNN':
        # just replace the model's path
        model_ft = Ensemble(['./ensemble/RES.pth', './ensemble/VGG.pth'], num_classes=15)
        # crop size
        input_size = 224
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    optimizer_ft = optim.Adam(params_to_update, lr=0.001)

    return model_ft, optimizer_ft, input_size


def test_model(model, input_size, test_dir):
    batch_size = 8
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
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=N_worker,
                                             drop_last=True)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            # input, target = batch
            input = batch[0].to(device)
            target = batch[1].to(device)
            output = model(input)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += torch.sum(predicted == target.data)
    #             if i % 100 == 99:
    #                 print(i)
    #                 print('batch: %5d,\t acc: %f' % (i + 1, correct / total))
    # print('--test accu: {}'.format(100.0 * correct / len(testloader.dataset)))
    print(float(100.0 * correct / len(testloader.dataset)))
    return float(100.0 * correct / len(testloader.dataset))


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
    image_datasets = {x: datasets.ImageFolder(os.path.join(train_data_dir, x), data_transforms[x]) for x in
                      ['train', 'test']}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                                       num_workers=N_worker) for x in ['train', 'test']}
    # train & val
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
                                 optimizer_ft, num_epochs)
    # save
    torch.save(model_ft.state_dict(), model_dir)
    print('model saved.')


def test(test_dir, load_model_path):
    print('Test...')
    model_ft, _, input_size = initialize_model()
    # load parameters(only, no structure)
    ######## Note: if using CPU, must declare map_location
    model_ft.load_state_dict(torch.load(load_model_path, map_location=device))
    print('checkpoint loaded.')
    acc = test_model(model_ft, input_size, test_dir)
    return acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='test', choices=['train', 'test'])
    ##! it is important that the dataset should include train set and validation set
    parser.add_argument('--train_data_dir', default='./dataset/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./dataset/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='./load_model/ensemble.pth', help='the pre-trained model')
    opt = parser.parse_args()
    # opt = parser.parse_args(args=[]) if gpu, pls replace with it

    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)
