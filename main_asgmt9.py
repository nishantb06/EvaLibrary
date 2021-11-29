

from models.models_Asgmt9 import CustomResNet
import torch.optim as optim
from src.dataloader_Asgmt9 import trainloader,testloader,classes
import torch
from src.train import train
from src.test import test
import torch.nn as nn
from utils.gradcam import *
from utils.visualize import *
from utils.identify_image import *
from utils.misclassified import *
from utils.denormalization import *

device  = "cuda"

def engine():
    model = CustomResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), 
                                                    epochs=24,pct_start = 5/24, verbose = True, anneal_strategy = 'linear',
                                                    div_factor = 10,three_phase = False,base_momentum = 0.8)

    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for epoch in range(1, 2):

        print(f"Epoch {epoch}")
        

        x,y = train(model, device, train_loader = trainloader, optimizer = optimizer,epoch =  epoch)
        
        a,b = test(model, device, test_loader = testloader)
        scheduler.step()
        

        train_losses.append(x)
        test_losses.append(a)
        train_accuracies.append(y)
        test_accuracies.append(b)
    
    return model,train_accuracies,train_losses,test_accuracies,test_losses

model,train_accuracies,train_losses,test_accuracies,test_losses = engine()

criterion = nn.CrossEntropyLoss()
correct_images, incorrect_images = identify_images(model, criterion, device, testloader, 10)
plot_images(incorrect_images, classes)

target_layers = ["layer3"]
viz_cam = VisualizeCam(model,classes, target_layers)
num_img = 20

incorrect_pred_imgs = []
for i in range(10):
  incorrect_pred_imgs.append(torch.as_tensor(incorrect_images[i]["img"]))
viz_cam(torch.stack(incorrect_pred_imgs), target_layers, metric="incorrect")
