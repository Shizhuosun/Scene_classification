import torch.nn as nn
import os
import string
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as f
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
# Transform the data
data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.Grayscale(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   # transforms.Grayscale(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = '../dataset'
train_root = data_root + '/train'
test_root = data_root + '/test'
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

test_dataset = datasets.ImageFolder(root=test_root,transform=data_transform["test"])

test_num = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=4, shuffle=True,
                                                  num_workers=0)

print("using {} images for training, {} images for validation.".format(train_num,test_num))

from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

encoder = EncoderCNN(512).to(device)
decoder = DecoderRNN(512, 1024, 15, 1).to(device)
# net = CnnLstm()
# net.to(device)
loss_function = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters())# + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.0001)
# pata = list(net.parameters())
# optimizer = optim.Adam(net.parameters(), lr=0.0001)

epochs = 10
save_path = '../CNN_RNN2.pth'
best_acc = 0.0
# train_steps = len(train_loader)
# for epoch in range(epochs):
#     # train
#     net.train()
#     running_loss = 0.0
#     train_bar = tqdm(train_loader)
#     for step, data in enumerate(train_bar):
#         images, labels = data
#         optimizer.zero_grad()
#         outputs = net(images.to(device))
#         loss = loss_function(outputs, labels.to(device))
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#
#         train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
#                                                                      epochs,
#                                                                      loss)
#
#     # validate
#     net.eval()
#     acc = 0.0  # accumulate accurate number / epoch
#     with torch.no_grad():
#         val_bar = tqdm(test_loader)
#         for val_data in val_bar:
#             val_images, val_labels = val_data
#             outputs = net(val_images.to(device))
#             predict_y = torch.max(outputs, dim=1)[1]
#             acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
#
#     val_accurate = acc / test_num
#     print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
#               (epoch + 1, running_loss / train_steps, val_accurate))
#
#     if val_accurate > best_acc:
#         best_acc = val_accurate
#         torch.save(net.state_dict(), save_path)
#
# print('Finished Training')
num_epochs = 10
model_path = '../CNN_RNN2.pth'
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, captions, lengths) in enumerate(train_loader):

        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

            # Save the model checkpoints
        if (i + 1) % 1000 == 0:
            torch.save(decoder.state_dict(), os.path.join(
                model_path, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            torch.save(encoder.state_dict(), os.path.join(
                model_path, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
