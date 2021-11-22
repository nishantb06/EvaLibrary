import matplotlib.pyplot as plt
import numpy as np
import torch

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def get_statistics(dataset):

    """
    input = dataset should be a of type torchvision.datasets calculate statistics of training data only 
    Calculated stats are->
    torch.Size([3, 32, 32, 50000])
    Mean =  tensor([0.4914, 0.4822, 0.4465])
    Standard deviation =  tensor([0.2470, 0.2435, 0.2616])
    
    """
    imgs = torch.stack([img_t for img_t,_ in dataset],dim = 3)
    print(imgs.shape)

    mean_calc = imgs.view(3,-1).mean(dim = 1)
    std_dev_calc = imgs.view(3,-1).std(dim = 1)

    print('Mean = ',mean_calc)
    print('Standard deviation = ',std_dev_calc)

    return mean_calc,std_dev_calc