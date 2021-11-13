import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm


"""
Guideline of your submission of HW3.
If you have any questions in regard to submission,
please contact TA via e0444157@nus.edu.sg
"""
###################################### Main train and test Function ####################################################
"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the 
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training 
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test, 
computing accuracy, etc.
"""


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def train(train_data_dir, model_dir, **kwargs):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
    data_root = './dataset'
    train_root = train_data_dir
    test_root = data_root + '/test'

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # Transform the data
    train_dataset = datasets.ImageFolder(root=train_root, transform=data_transform["train"])
    train_num = len(train_dataset)
    print(train_num)
    scene_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in scene_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    test_dataset = datasets.ImageFolder(root=test_root, transform=data_transform["test"])

    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=4, shuffle=True,
                                              num_workers=0)
    print("start training.............")
    net = AlexNet(num_classes=15).to(device)
    # load model weights
    # model_weight_path = model_dir
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    epochs = 10
    # save_path = './{}Net.pth'.format(model_name)
    save_path = model_dir
    best_acc = 0.0
    train_steps = len(train_loader)
    total = 0
    correct = 0
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            predict = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += sum((predict == labels)).item()
            if step % 200 == 199:  # every 200 steps
                print(
                    'epoch %5d: batch: %5d, loss: %f  acc %f' % (epoch + 1, step + 1, running_loss / 200, correct / total))
                running_loss = 0.0
    print('Accuracy: %.2f%%' % (correct / total * 200))
    torch.save(net.state_dict(), save_path)
    print('Finished Training')




def test(test_data_dir, model_dir, **kwargs):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """
    data_root = './dataset'
    train_root = data_root + '/train'
    test_root = test_data_dir

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # Transform the data
    train_dataset = datasets.ImageFolder(root=train_root, transform=data_transform["train"])
    train_num = len(train_dataset)
    print(train_num)
    scene_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in scene_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    test_dataset = datasets.ImageFolder(root=test_root, transform=data_transform["test"])

    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=4, shuffle=True,
                                              num_workers=0)
    print("start test............")

    net = AlexNet(num_classes=15).to(device)
    # load model weights
    model_weight_path = model_dir
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        # val_bar = tqdm(test_loader)
        for i, val_data in enumerate(test_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            total += val_labels.size(0)
            correct += torch.eq(predict_y, val_labels.to(device)).sum().item()
            if i % 100 == 99:
                print('batch: %5d,\t acc: %f' % (i + 1, correct / total))
    print('Accuracy: %.2f%%' % (correct / total * 100))
    print("Finish testing!")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='test', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./dataset/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./dataset/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='./CNN_AlexNet.pth', help='the pre-trained model')
    opt = parser.parse_args()


    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)
