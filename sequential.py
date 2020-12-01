# from __future__ import print_function
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import fastbook
fastbook.setup_book()
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
from fastai.vision.all import *
from fastbook import *
transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()
path = untar_data('https://s3.amazonaws.com/fast-ai-imageclas/mnist_png.tgz')

zeros, ones, twos, threes, fours, fives, sixs, sevens, eights, nines = [tensor(Image.open(o)) for o in (path/'testing/0').ls()], [tensor(Image.open(o)) for o in (path/'testing/1').ls()],[tensor(Image.open(o)) for o in (path/'testing/2').ls()], [tensor(Image.open(o)) for o in (path/'testing/3').ls()],[tensor(Image.open(o)) for o in (path/'testing/4').ls()], [tensor(Image.open(o)) for o in (path/'testing/5').ls()],[tensor(Image.open(o)) for o in (path/'testing/6').ls()], [tensor(Image.open(o)) for o in (path/'testing/7').ls()],[tensor(Image.open(o)) for o in (path/'testing/8').ls()], [tensor(Image.open(o)) for o in (path/'testing/9').ls()]
# for image in (path/'testing').ls().sorted():
    # images.append([tensor(Image.open(o)) for o in image])
dset = list(zip(zeros,ones,twos,threes,fours,fives,sixs,sevens,eights,nines))
# print()
print((path/'testing').ls())
dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('../data', train=False, download=True,
                    transform=transform)
# print(list(DataLoader(datasets.MNIST('../data', transform=transform))))
trainloader = DataLoader((path/'testing').ls(), batch_size=512, shuffle=True)
testloader = DataLoader((path/'testing').ls(), batch_size=1024, shuffle=True)
dl = DataLoader(dset, batch_size=64)
x =  first(dl)
print(x, len(x))
dls = DataLoaders(trainloader, testloader)
# images, labels = next(iter(trainloader))
# images = images.view(-1, 28*28  )
# criterion = nn.CrossEntropyLoss()
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),        
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10))
learn = Learner(dset, model, opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(10, lr=1)

# logits = model(images)
# loss = criterion(logits, labels)
# print(loss)