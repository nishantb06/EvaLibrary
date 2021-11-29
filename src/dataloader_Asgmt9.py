import cv2
import torchvision
import torch
import torchvision.transforms as transforms

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
device = 'cuda'
torch.manual_seed(2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

train_transforms = A.Compose(
    [
        A.augmentations.transforms.PadIfNeeded(40,40),
        A.augmentations.transforms.RandomCrop(32,32),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=8, min_width=8,
                        fill_value=0.4734),
        A.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616),
            p =1.0
        ),
        ToTensorV2()
    ],
    p=1.0
)

test_transforms = A.Compose(
    [
        A.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616),
            p =1.0
        ),
        ToTensorV2()
    ]
)

class args():
    def __init__(self,device = 'cpu' ,use_cuda = False) -> None:
        self.batch_size = 512
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 2, 'pin_memory': True} if self.use_cuda else {}

trainset = Cifar10SearchDataset(root='./data', train=True,
                                        download=True, transform=train_transforms)
                                        
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args().batch_size,
                                          shuffle=True, **args(use_cuda=True).kwargs)


testset = Cifar10SearchDataset(root='./data', train=False,
                                       download=True, transform=test_transforms)

testloader = torch.utils.data.DataLoader(testset, batch_size=args().batch_size,
                                         shuffle=False, **args(use_cuda=True).kwargs)