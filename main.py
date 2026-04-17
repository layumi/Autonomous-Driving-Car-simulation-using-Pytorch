import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import csv
import cv2
from tqdm import tqdm
from train_model import DriverNet, ft_resnet18
import argparse
from torchvision import transforms

np.random.seed(0)
sample = []

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
parser.add_argument('--data_dir', help='data directory', type=str, default='./images')
parser.add_argument('--test_size', help='test size fraction', type=float, default=0.2)
parser.add_argument('--keep_prob', help='drop out probability', type=float, default=0.5)
parser.add_argument('--nb_epoch', help='number of epochs', type=int, default=10)
parser.add_argument('--samples_per_epoch', help='samples per epoch', type=int, default=20000)
parser.add_argument('--batchsize', help='batch size', type=int, default=128)
parser.add_argument('--lr_rate', help='learning rate', type=float, default=3.5e-4)
opt = parser.parse_args() 

with open(opt.data_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        sample.append(line)


def augment(imgName, angle):
    name = imgName.split('\\')[-1]
    current_image = cv2.imread(name)
    current_image = current_image[65:-25, :, :]
    if np.random.rand() < 0.5:
        current_image = cv2.flip(current_image, 1)
        angle = angle * -1.0
    return current_image, angle


train_len = int(0.8 * len(sample))
valid_len = len(sample) - train_len
train_samples, validation_samples = data.random_split(sample, lengths=[train_len, valid_len])


class Dataset(data.Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index]
        steering_angle = float(batch_samples[3])
        center_img, steering_angle_center = augment(batch_samples[0], steering_angle)
        left_img, steering_angle_left = augment(batch_samples[1], steering_angle + 0.4)
        right_img, steering_angle_right = augment(batch_samples[2], steering_angle - 0.4)
        center_img = self.transform(center_img)
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)
        return (center_img, steering_angle_center), (left_img, steering_angle_left), (right_img, steering_angle_right)

    def __len__(self):
        return len(self.samples)



def _my_normalization(x):
    return x / 127.5 - 1.0


transformations = transforms.Compose([transforms.Lambda(_my_normalization)])

training_set = Dataset(train_samples, transformations)
training_generator = DataLoader(training_set, batch_size=opt.batchsize, shuffle=True)

validation_set = Dataset(validation_samples, transformations)
validation_generator = DataLoader(validation_set,  batch_size=opt.batchsize, shuffle=False)


def build_model():
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 4x4, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    #model = DriverNet()
    model = ft_resnet18()
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def toDevice(datas, device):
    imgs, angles = datas
    return imgs.float().to(device), angles.float().to(device)


epochs = 22


def train_model(model):
    """
    Train the model
    """
    optimizer = optim.Adam(model.parameters(), lr=opt.lr_rate)
    criterion = nn.MSELoss()
    model.to(device)
    for epoch in range(epochs):
        # Training
        train_loss = 0
        train_total = 0
        model.train()
        for local_batch, (centers, lefts, rights) in tqdm(enumerate(training_generator)):
            centers, lefts, rights = toDevice(centers, device), toDevice(lefts, device), toDevice(rights, device)
            optimizer.zero_grad()
            datas = [centers, lefts, rights]
            for data in datas:
                imgs, angles = data
                outputs = model(imgs)
                loss = criterion(outputs, angles.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_total += 1

        model.eval()
        valid_loss = 0
        valid_total = 0
        with torch.set_grad_enabled(False):
            for local_batch, (centers, lefts, rights) in tqdm(enumerate(validation_generator)):
                centers, lefts, rights = toDevice(centers, device), toDevice(lefts, device), toDevice(rights, device)
                optimizer.zero_grad()
                datas = [centers, lefts, rights]
                for data in datas:
                    imgs, angles = data
                    outputs = model(imgs)
                    loss = criterion(outputs, angles.unsqueeze(1))

                    valid_loss += loss.item()
                    valid_total += 1
        print('epoch:{} \t train_loss: {} \t valid_loss:{}'.format(epoch, train_loss / train_total,
                                                                   valid_loss / valid_total))
    torch.save(model, 'model.h5')



# build model
model = build_model()
# train model on data, it saves as model.h5
train_model(model)


