import torch.nn.functional as F
from EvaLibrary.src.dataloader import trainloader
from tqdm import tqdm
import torch.nn as nn

def train(model, device,optimizer, epoch, train_loader):

    model.train() #set  the model  in training mode (which means the model knows to include thing like batchnorm and dropout)
    pbar = tqdm(train_loader)
    correct  = 0
    processed  = 0
    train_acc = []
    train_losses = []
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        # get the inputs
        data = data.to(device)
        target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        #printing training logs
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        train_losses.append(loss.item())

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

        # if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, batch_idx + 1, running_loss / 2000))
        #     running_loss = 0.0


    return train_losses,train_acc