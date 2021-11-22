from torchvision.utils import make_grid
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

def identify_images(net, criterion, device, testloader, n):
    net.eval()
    correct_images = []
    incorrect_images = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)           
            predicted = outputs.argmax(dim=1, keepdim=True)
            is_correct = predicted.eq(targets.view_as(predicted))
            
            misclassified_inds = (is_correct==0).nonzero()[:,0]
            for mis_ind in misclassified_inds:
              if len(incorrect_images) == n:
                break
              incorrect_images.append({
                  "target": targets[mis_ind].cpu().numpy(),
                  "pred": predicted[mis_ind][0].cpu().numpy(),
                  "img": inputs[mis_ind]
              })

            correct_inds = (is_correct==1).nonzero()[:,0]
            for ind in correct_inds:
              if len(correct_images) == n:
                break
              correct_images.append({
                  "target": targets[ind].cpu().numpy(),
                  "pred": predicted[ind][0].cpu().numpy(),
                  "img": inputs[ind]
              })
    return correct_images, incorrect_images
  
  
def plot_images(img_data, classes):
    figure = plt.figure(figsize=(10, 10))

    num_of_images = len(img_data)
    for index in range(1, num_of_images + 1):
        img = denormalize(img_data[index-1]["img"])  # unnormalize
        plt.subplot(5, 5, index)
        plt.axis('off')
        img = img.cpu().numpy()
        maxValue = np.amax(img)
        minValue = np.amin(img)
        img = np.clip(img, 0, 1)
        img = img/np.amax(img)
        img = np.clip(img, 0, 1)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title("Predicted: %s\nActual: %s" % (classes[img_data[index-1]["pred"]], classes[img_data[index-1]["target"]]))

    plt.tight_layout()