from tqdm import tqdm
import torch
import torch.nn.functional as F
from dataloader import testloader
from dataloader import trainloader

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
        loss = F.nll_loss(output, target)
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


def test(model, device, test_loader):

    model.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    test_acc = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target,reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100*correct/len(test_loader.dataset))
    
    return test_losses,test_acc