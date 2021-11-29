# print albumentations
# !pip install albumentations

from EvaLibrary.models.models_Asgmt9 import CustomResNet
import torch.optim as optim
from EvaLibrary.src.dataloader_Asgmt9 import trainloader,testloader
import torch
from EvaLibrary.src.train import train
from EvaLibrary.src.test import test

device  = "cuda"

model = CustomResNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), 
                                                epochs=24,pct_start = 5/24, verbose = True, anneal_strategy = 'linear',
                                                div_factor = 10,three_phase = False,base_momentum = 0.8)

train_accuracies = []
train_losses = []
test_accuracies = []
test_losses = []

for epoch in range(1, 25):

    print(f"Epoch {epoch}")
    

    x,y = train(model, device, train_loader = trainloader, optimizer = optimizer,epoch =  epoch)
    
    a,b = test(model, device, test_loader = testloader)
    scheduler.step()
    

    train_losses.append(x)
    test_losses.append(a)
    train_accuracies.append(y)
    test_accuracies.append(b)
